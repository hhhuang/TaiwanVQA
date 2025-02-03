# TaiwanVQA

[[**Project Page**](https://github.com/hhhuang/TaiwanVQA)] | [[**🤗 TaiwanVQA**](https://huggingface.co/datasets/hhhuang/TaiwanVQA)] | [[**TaiwanVQA arXiv**](https://github.com/hhhuang/TaiwanVQA)]



This repository provides an evaluation script and configuration for the **TaiwanVQA** benchmark—*TaiwanVQA: A Visual Question Answering Benchmark for Taiwan-Specific Content*—based on a public multimodal evaluation framework (see [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)). It is designed to benchmark large vision-language models on TaiwanVQA.

---

## Repository Structure

```
TaiwanVQA-lmms-eval
├── README.md
├── eval_taiwanvqa.sh
└── lmms-eval
    ├── models
    │   ├── __init__.py      # Register the custom "vllm_api" model
    │   └── vllm_api.py
    └── tasks
        └── taiwanvqa
            ├── taiwanvqa_evals.py
            ├── taiwanvqa.yaml
            └── utils.py
```


## How to Evaluate with TaiwanVQA

### 1. Host Your Model with vLLM

Before evaluation, you must host your model using [vLLM](https://github.com/vllm-project/vllm).   You can install it via:

```bash
pip install vllm
```

Then,run the model service. For example:

```bash
vllm serve microsoft/Phi-3.5-vision-instruct \
    --task generate \
    --trust-remote-code \
    --max-model-len 8192 \
    --limit-mm-per-prompt image=1 \
    --dtype half \
    --tensor-parallel-size 8 \
    --gpu_memory_utilization 0.9 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 8 \
    --enforce-eager \
    --swap-space 1
```

This will start a RESTful API, typically at http://localhost:8000/v1.

If you are hosting the model in a container, you can check the container’s IP address by running:
```
HOSTIP=$(hostname -i)
echo $HOSTIP
```

Then, replace localhost in the API URL with the container’s IP (e.g., http://<container-ip>:8000/v1).


### 2. Install the Evaluation Framework

Please refer to the official lmms-eval installation instructions available [here](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main?tab=readme-ov-file#installation).  
After installing `lmms-eval` (via PyPI or by cloning the repository), add the TaiwanVQA task files provided in this repository as follows:
* Copy the files under the `models` folder in this repository (i.e., `vllm_api.py` and the necessary update to `__init__.py`) to the `lmms-eval`'s `models` folder.
    * **Note**: You do not need to copy the entire `__init__.py` from this repository. Instead, just add the following entry to the `AVAILABLE_MODELS` dictionary in your  `lmms-eval`'s `__init__.py` file (if it's not already present):
        ```python
        "vllm_api": "vLLM_API",
        ```

* Copy the entire `taiwanvqa` folder (containing `taiwanvqa_evals.py`, `taiwanvqa.yaml`, and `utils.py`) to the `lmms-eval`'s `tasks` folder.

Also, follow the [lmms-eval's task guide](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/task_guide.md) for any additional integration details.


### 3. Run the Evaluation

With your model service up and the evaluation framework installed, run the evaluation with the following command:

```
python3 -m accelerate.commands.launch \
    --num_processes 8 \
    -m lmms_eval \
    --model vllm_api \
    --model_args model_version=Qwen/Qwen2-VL-7B-Instruct,modality=image,base_url=http://localhost:8000/v1 \
    --tasks taiwanvqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix taiwanvqa \
    --output_path ./results/taiwanvqa/
```

**Explanation of key parameters:**
* `--model vllm_api`: Uses the custom model class that communicates with your vLLM-hosted service.
* `--model_args`:
    * `model_version`: Specifies the version or name of the model.
    * `modality`: Indicates the model modality (e.g., `image`).
    * `base_url`: The URL where your vLLM service is hosted.
        * Note: If hosting on a container, replace `http://localhost:8000/v1` with `http://<container-ip>:8000/v1` (use the container’s IP from the `hostname -i` command).
* `--tasks taiwanvqa`: Runs the TaiwanVQA evaluation task.
* `--output_path`: Directory where evaluation results are saved.

## References
* [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
* [vLLM](https://github.com/vllm-project/vllm)


## Citation

If you use TaiwanVQA in your research, please cite as follows:

```
@inproceedings{taiwanvqa,
  title={TaiwanVQA: A Visual Question Answering Benchmark for Taiwanese Daily Life},
  author={Hsin-Yi Hsieh, Shang-Wei Liu, Chang-Chih Meng, Chien-Hua Chen, Shuo-Yueh Lin, Hung-Ju Lin, Hen-Hsen Huang, I-Chen Wu},
  booktitle={EvalMG25 @ COLING 2025},
  year={2024}
}
```