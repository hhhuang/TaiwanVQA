import base64
import json
import os
import time
import gc
import torch
from datetime import datetime
from io import BytesIO
from typing import List, Tuple

import numpy as np
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from PIL import Image

from openai import OpenAI
from PIL import Image
from typing import Union


NUM_SECONDS_TO_SLEEP = 10
from loguru import logger as eval_logger

@register_model("vllm_api")
class vLLM_API(lmms):
    def __init__(
        self,
        model_version: str = "gpt-4o",
        modality: str = "image",
        base_url: str = None,
        api_key: str = "EMPTY", 
        max_frames_num: int = 10,
        timeout: int = 120,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for the vlm so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient

       
        if base_url is None:
            raise ValueError("`base_url` must not be None. Please specify a valid URL.")
        
        self.base_url = base_url
        self.api_key = api_key

        self.model_version = model_version
        self.modality = modality
        self.max_frames_num = max_frames_num
        self.image_token = "<image>"
        self.timeout = timeout
        self.continual_mode = continual_mode

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")

            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

    # Function to encode the image
    def encode_image(self, image: Image):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    
    def extract_logprobs(self, response, target_tokens=["A", "B", "C", "D"]):
        """
        Extract logprob information using actual logprob values.
        
        Args:
            response: The OpenAI API response object
            target_tokens: List of tokens to analyze (default ["A", "B", "C", "D"])
            
        Returns:
            dict: Dictionary containing:
                - result_token: The token among A,B,C,D with highest logprob
                - token_order: Position of result_token in sorted alternatives
                - token_logprobs: Actual logprob for the result token
                - options_logprob: Dictionary mapping A, B, C, D to their actual logprobs
                - top_alternatives: Flat dictionary of all alternative tokens and their logprobs
        """
        # Get the response content
        choices = response.choices[0]
        
        # Check if logprobs are available
        if not hasattr(choices, 'logprobs') or choices.logprobs is None:
            return None
        
        # Initialize result structures
        result = {
            'result_token': None,
            'token_order': None,
            'token_logprobs': {},
            'options_logprob': {},
            'top_alternatives': {}
        }
        
        # Process tokens to get alternatives and their logprobs
        for token_info in choices.logprobs.content:
            # Store all alternatives
            if hasattr(token_info, 'top_logprobs'):
                # Create flat alternatives dictionary including the current token
                result['top_alternatives'] = {
                    alt.token: alt.logprob 
                    for alt in token_info.top_logprobs
                }
                result['top_alternatives'][token_info.token] = token_info.logprob
                
                # Get logprobs for all target tokens (A, B, C, D)
                for target in target_tokens:
                    if target in result['top_alternatives']:
                        result['options_logprob'][target] = result['top_alternatives'][target]
                    else:
                        result['options_logprob'][target] = float('-inf')
                
                # Find the target token with highest logprob
                max_logprob = float('-inf')
                for token in target_tokens:
                    if result['options_logprob'][token] > max_logprob:
                        max_logprob = result['options_logprob'][token]
                        result['result_token'] = token
                
                if result['result_token']:
                    # Set token_logprobs with actual logprob value
                    result['token_logprobs'] = result['options_logprob'][result['result_token']]
                    
                    # Sort all alternatives by logprob to find the position of result_token
                    sorted_tokens = sorted(
                        result['top_alternatives'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # Find position of result_token in sorted list
                    for idx, (token, _) in enumerate(sorted_tokens):
                        if token == result['result_token']:
                            result['token_order'] = idx
                            break
                break
        
        return result

    def generate_until(self, requests) -> List[str]:
        res = []
        total_requests = len(requests)
        start_time = datetime.now()
        
        # Initialize memory tracking
        initial_memory = {i: torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())}
        last_memory_check = time.time()
        MEMORY_CHECK_INTERVAL = 300  # Check memory every 5 minutes
        MEMORY_THRESHOLD_GB = 30  # Alert threshold for 32GB GPUs
        
        # Progress bar
        pbar = tqdm(total=total_requests, disable=(self.rank != 0), desc="Model Responding")
        
        def cleanup_memory():
            """Clean up GPU memory without restart"""
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)  # Give some time for memory to be freed
            
        batch_size = 4  # Start with a conservative batch size
        for batch_idx in range(0, len(requests), batch_size):
            current_batch = requests[batch_idx:batch_idx + batch_size]
            batch_results = []
            
            # Memory check and cleanup if needed
            current_time = time.time()
            if current_time - last_memory_check > MEMORY_CHECK_INTERVAL:
                last_memory_check = current_time
                for device in range(torch.cuda.device_count()):
                    current_memory = torch.cuda.memory_allocated(device)
                    memory_gb = current_memory / (1024**3)
                    
                    if memory_gb > MEMORY_THRESHOLD_GB:
                        eval_logger.warning(f"High memory usage on GPU {device}: {memory_gb:.2f}GB. Running cleanup...")
                        cleanup_memory()
                        
                        # Log memory after cleanup
                        new_memory_gb = torch.cuda.memory_allocated(device) / (1024**3)
                        eval_logger.info(f"After cleanup: GPU {device} memory: {new_memory_gb:.2f}GB")

            # Process each request in the batch
            for reg in current_batch:
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = reg.args
                
                # Check cache for continual mode
                if self.continual_mode and self.cache_mode == "resume":
                    doc_uuid = f"{task}___{split}___{doc_id}"
                    if doc_uuid in self.response_cache:
                        response_text = self.response_cache[doc_uuid]
                        if response_text:
                            batch_results.append(response_text)
                            continue

                try:
                    if self.modality == "text":
                        messages = [{"role": "user", "content": [{"type": "text", "text": contexts}]}]
                    else:
                        # Process visuals
                        visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                        visuals = self.flatten(visuals)
                        imgs = []

                        # Process images/frames with memory cleanup
                        for visual in visuals:
                            if self.modality == "image":
                                img = self.encode_image(visual)
                                imgs.append(img)
                            elif self.modality == "video":
                                frames = self.encode_video(visual, self.max_frames_num)
                                imgs.extend(frames)
                        
                        # Free some memory after image processing
                        if not hasattr(self, '_image_cache'):
                            cleanup_memory()

                        # Prepare messages
                        messages = []
                         # When there is no image token in the context, append the image to the text
                        if self.image_token not in contexts:
                            content = [{"type": "text", "text": contexts}]
                            for img in imgs:
                                content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img}"}
                                })
                            messages.append({"role": "user", "content": content})
                        else:
                            contexts_split = contexts.split(self.image_token)
                            for idx, img in enumerate(imgs):
                                content = [
                                    {"type": "text", "text": contexts_split[idx]},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                                ]
                                messages.append({"role": "user", "content": content})

                            # If n image tokens are in the contexts
                            # contexts will be splitted into n+1 chunks
                            # Manually add it into the payload
                            if contexts_split[-1]:
                                messages.append({
                                    "role": "user",
                                    "content": [{"type": "text", "text": contexts_split[-1]}]
                                })

                    # Set generation parameters
                    max_tokens = gen_kwargs.get("max_new_tokens", 10)
                    temperature = gen_kwargs.get("temperature", 0)

                    # API call with retries
                    response_text = ""
                    max_retries = 3
                    retry_delay = 5  # seconds
                    
                    for attempt in range(max_retries):
                        try:
                            response = self.client.chat.completions.create(
                                model=self.model_version,
                                messages=messages,
                                logprobs=True,
                                top_logprobs=20,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                timeout=self.timeout
                            )

                            results = self.extract_logprobs(response, target_tokens=["A", "B", "C", "D", " A", " B", " C", " D"])
                            print(f'results: {results}')
                            
                            response_text = results.get('result_token', "")
                            response_text = response_text.strip() if response_text else ""

                            break  # Success, exit retry loop
                            
                        except Exception as e:
                            eval_logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                            if attempt < max_retries - 1:
                                cleanup_memory()  # Clean memory before retry
                                time.sleep(retry_delay)
                            else:
                                eval_logger.error(f"All {max_retries} attempts failed for request {doc_id}")
                                response_text = ""

                    batch_results.append(response_text)

                    # Cache response if in continual mode
                    if self.continual_mode:
                        doc_uuid = f"{task}___{split}___{doc_id}"
                        self.response_cache[doc_uuid] = response_text
                        with open(self.response_persistent_file, "w") as f:
                            json.dump(self.response_cache, f)

                except Exception as e:
                    eval_logger.error(f"Error processing request {doc_id}: {str(e)}")
                    batch_results.append("")  # Add empty result on error
                    cleanup_memory()  # Clean memory after error

            # Extend results with batch results
            res.extend(batch_results)
            pbar.update(len(current_batch))

            # Progress and memory logging
            if batch_idx > 0 and batch_idx % (batch_size * 5) == 0:
                elapsed = datetime.now() - start_time
                processed = batch_idx + len(current_batch)
                rate = processed / elapsed.total_seconds()
                remaining = (total_requests - processed) / rate if rate > 0 else 0
                
                # Log progress and memory usage
                for device in range(torch.cuda.device_count()):
                    current_memory = torch.cuda.memory_allocated(device)
                    diff = current_memory - initial_memory[device]
                    eval_logger.info(
                        f"Progress: {processed}/{total_requests} requests. "
                        f"Rate: {rate:.2f} req/s. "
                        f"Estimated remaining time: {remaining/3600:.2f} hours. "
                        f"GPU {device} memory change: {diff/1024**3:.2f}GB"
                    )

                # Optional: force cleanup after logging
                if diff / (1024**3) > 2:  # If memory increased by more than 2GB
                    cleanup_memory()

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for vllm api")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "vllm api not support"
