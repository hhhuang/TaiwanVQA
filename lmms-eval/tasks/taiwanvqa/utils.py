import json
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.taiwanvqa.taiwanvqa_evals import TaiwanVQA_Evaluator

with open(Path(__file__).parent / "taiwanvqa.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


taiwanvqa_evaluator = TaiwanVQA_Evaluator(sys_prompt=config["metadata"]["sys_prompt"])


def taiwanvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def taiwanvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_candidate = ["A", "B", "C", "D"]
    options_prompt, options_dict = taiwanvqa_evaluator.create_options_prompt(doc, option_candidate)

    data = {
        "question": doc["question"],
        "answer": doc.get("answer", None),
        "options": options_prompt,
        "sub_topic": doc["sub_topic"],
        "topic": doc["topic"],
        "options_dict": options_dict,
        "question_id": doc["question_id"],
    }

    query_prompt = f"{data['question']} {data['options']}"

    if lmms_eval_specific_kwargs:
        query_prompt = f"{query_prompt}\n{lmms_eval_specific_kwargs['post_prompt']}"

    print(f"Query Prompt: {query_prompt}")
    return query_prompt


def taiwanvqa_process_results(doc, results):
    model_response = results[0].strip()

    # Get the correct answer
    gold = doc["answer"]

    # Compute accuracy
    acc = 1.0 if model_response == gold else 0.0

    data = {
        "gpt_eval_score": {
            "question_id": doc["question_id"],
            "image_id": doc["image_id"],
            "acc": acc,
            "question": doc["question"],
            "answer": doc["answer"],
            "prediction": model_response,
            "question_type": doc["question_type"],
            "topic": doc["topic"],
            "sub_topic": doc["sub_topic"],
            "location": doc["location"],
            "difficulty_level": doc["difficulty_level"],
            "is_ocr": doc["is_ocr"],
        },
        "submission": {
            "question_id": doc["question_id"],
            "image_id": doc["image_id"],
            "acc": acc,
            "question": doc["question"],
            "answer": doc["answer"],
            "prediction": model_response,
            "question_type": doc["question_type"],
            "topic": doc["topic"],
            "sub_topic": doc["sub_topic"],
            "location": doc["location"],
            "difficulty_level": doc["difficulty_level"],
            "is_ocr": doc["is_ocr"],
            "A": doc.get("A", "nan"),
            "B": doc.get("B", "nan"),
            "C": doc.get("C", "nan"),
            "D": doc.get("D", "nan"),
        },
    }
    option_candidate = ["A", "B", "C", "D"]
    for c in option_candidate:
        data["submission"][c] = doc.get(c, "nan")
        data["gpt_eval_score"][c] = doc.get(c, "nan")

    return data


def taiwanvqa_aggregate_results_eval(results, args):
    print(f"========== TaiwanVQA Evaluation Results ==========")
    eval_results = taiwanvqa_evaluator.eval_result(results)

    overall_acc = eval_results["overall_accuracy"]
    question_type_acc = eval_results["question_type_accuracies"]
    topic_acc = eval_results["topic_accuracies"]
    topic_question_type_acc = eval_results["topic_question_type_accuracies"]
    sub_topic_acc = eval_results["sub_topic_accuracies"]
    sub_topic_question_type_acc = eval_results["sub_topic_question_type_accuracies"]
    is_ocr_acc = eval_results["is_ocr_accuracies"]
    difficulty_level_acc = eval_results["difficulty_level_accuracies"]

    # Print overall accuracy
    print(f"Overall Accuracy (CircularEval): {overall_acc * 100:.2f}%")

    # Print accuracy per question type
    print("\nAccuracy per Question Type:")
    for q_type, acc in question_type_acc.items():
        print(f"  {q_type}: {acc * 100:.2f}%")

    # Print per topic accuracy
    print("\nAccuracy per Topic:")
    for topic, acc in topic_acc.items():
        print(f"  {topic}: {acc * 100:.2f}%")

    # Print per topic per question type accuracy
    for q_type, topic_accs in topic_question_type_acc.items():
        print(f"\nAccuracy per Topic of {q_type} questions:")
        for topic, acc in topic_accs.items():
            print(f"  {topic}: {acc * 100:.2f}%")

    # Print per sub-topic accuracy
    print("\nAccuracy per Sub-topic:")
    for sub_topic, acc in sub_topic_acc.items():
        print(f"  {sub_topic}: {acc * 100:.2f}%")

    # print per sub-topic per question type accuracy
    for q_type, sub_topic_accs in sub_topic_question_type_acc.items():
        print(f"\nAccuracy per Sub-topic of {q_type} questions:")
        for sub_topic, acc in sub_topic_accs.items():
            print(f"  {sub_topic}: {acc * 100:.2f}%")

    # For Recognition, print is_ocr accuracy
    if is_ocr_acc:
        print("\nAccuracy per is_ocr (Recognition):")
        for is_ocr_value, acc in is_ocr_acc.items():
            print(f"  is_ocr = {is_ocr_value}: {acc * 100:.2f}%")

    # For Reasoning, print difficulty_level accuracy
    if difficulty_level_acc:
        print("\nAccuracy per Difficulty Level (Reasoning):")
        for level, acc in difficulty_level_acc.items():
            print(f"  Difficulty Level {level}: {acc * 100:.2f}%")

    # Save the results
    file = generate_submission_file("taiwanvqa_results.json", args)
    details_info = {
        "overall_accuracy": overall_acc,
        "question_type_accuracies": question_type_acc,
        "topic_accuracies": topic_acc,
        "topic_question_type_accuracies": topic_question_type_acc,
        "sub_topic_accuracies": sub_topic_acc,
        "sub_topic_question_type_accuracies": sub_topic_question_type_acc,
        "is_ocr_accuracies": is_ocr_acc,
        "difficulty_level_accuracies": difficulty_level_acc,
    }
    with open(file, "w", encoding="utf-8") as f:
        json.dump(details_info, f, indent=2, ensure_ascii=False)
    eval_logger.info(f"Saved detailed results to {file}")

    return overall_acc * 100


def taiwanvqa_aggregate_results_submission(results, args):
    df = pd.DataFrame(results)
    excel_write_path = generate_submission_file("taiwanvqa_results.xlsx", args)
    with pd.ExcelWriter(excel_write_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    eval_logger.info(f"Saved results to {excel_write_path}")
