import pandas as pd


class TaiwanVQA_Evaluator:
    def __init__(self, sys_prompt="There are several options:"):
        self.sys_prompt = sys_prompt

    def create_options_prompt(self, row_data, option_candidate):
        available_keys = set(row_data.keys()) & set(option_candidate)
        options = {cand: row_data[cand] for cand in available_keys if row_data[cand]}
        sorted_options = dict(sorted(options.items()))
        options_prompt = f"{self.sys_prompt}\n"
        for key, item in sorted_options.items():
            if pd.notna(item) and item != "nan":
                options_prompt += f"{key}. {item}\n"
        return options_prompt.rstrip("\n"), sorted_options

    def eval_sub_data(self, sub_data, answer_map):
        lt = len(sub_data)
        GT, PRED = [], []
        for i in range(lt):
            item = sub_data.iloc[i]
            idx = item["question_id"]
            GT.append(answer_map[idx])
            PRED.append(item["prediction"].strip())
            if PRED[-1] != GT[-1]:
                return 0  # Return 0 if any permutation is incorrect
        return 1  # Return 1 if all permutations are correct

    def calculate_hit_rates(self, data):
        overall_hit_rate = data["hit"].mean()

        category_hit_rate = {}
        if "sub_topic" in data.columns:
            # Category-based hit rate
            category_hit_rate = data.groupby("sub_topic")["hit"].mean().to_dict()

        # l2-category based hit rate
        l2_category_hit_rate = {}
        if "topic" in data.columns:
            l2_category_hit_rate = data.groupby("topic")["hit"].mean().to_dict()

        # question based hit rate
        question_type_hit_rate = {}
        if "question_type" in data.columns:
            question_type_hit_rate = data.groupby("question_type")["hit"].mean().to_dict()

        # difficulty level based hit rate
        difficulty_level_hit_rate = {}
        if "difficulty_level" in data.columns:
            difficulty_level_hit_rate = data.groupby("difficulty_level")["hit"].mean().to_dict()

        # location based hit rate
        location_hit_rate = {}
        if "location" in data.columns:
            location_hit_rate = data.groupby("location")["hit"].mean().to_dict()

        # is_ocr based hit rate
        is_ocr_hit_rate = {}
        if "is_ocr" in data.columns:
            is_ocr_hit_rate = data.groupby("is_ocr")["hit"].mean().to_dict()

        return overall_hit_rate, category_hit_rate, l2_category_hit_rate, question_type_hit_rate, difficulty_level_hit_rate, location_hit_rate, is_ocr_hit_rate

    def eval_result(self, results):
        data = pd.DataFrame(results)
        data["question_id"] = data["question_id"].astype(int)
        data = data.sort_values(by="question_id")
        data["prediction"] = data["prediction"].astype(str)

        # Extract original question IDs
        data["original_question_id"] = data["question_id"] % int(1e6)

        # Initialize data structures
        data_main = data.copy()
        data_main["hit"] = 0  # Initialize hit column to zero

        # Map for ground truth answers using question_id (includes permutations)
        answer_map = dict(zip(data["question_id"], data["answer"]))

        # Collect unique original question IDs
        unique_question_ids = data["original_question_id"].unique()

        total_questions = len(unique_question_ids)
        correct_questions = 0

        # Initialize a result dictionary to store evaluation outcomes
        result = {}

        # Loop over each unique original question ID
        for idx in unique_question_ids:
            # Group all permutations of the question
            sub_data = data[data["original_question_id"] == idx]

            # Apply CircularEval strategy
            ret = self.eval_sub_data(sub_data, answer_map)
            result[idx] = ret
            data_main.loc[data_main["original_question_id"] == idx, "hit"] = ret

            if ret == 1:
                correct_questions += 1

        # Compute overall accuracy
        overall_hit_rate = correct_questions / total_questions
        print(f"Total Questions: {total_questions} | Correct Questions: {correct_questions} | Overall Hit Rate: {overall_hit_rate}")

        # Compute accuracies based on question_type
        question_type_accuracies = {}
        for q_type in data_main["question_type"].unique():
            qtype_data = data_main[data_main["question_type"] == q_type]
            qtype_correct = qtype_data[qtype_data["hit"] == 1]["original_question_id"].nunique()
            qtype_total = qtype_data["original_question_id"].nunique()
            qtype_hit_rate = qtype_correct / qtype_total
            question_type_accuracies[q_type] = qtype_hit_rate

        # Compute accuracies based on topic
        topic_accuracies = {}
        for topic in data_main["topic"].unique():
            topic_data = data_main[data_main["topic"] == topic]
            topic_correct = topic_data[topic_data["hit"] == 1]["original_question_id"].nunique()
            topic_total = topic_data["original_question_id"].nunique()
            topic_hit_rate = topic_correct / topic_total
            topic_accuracies[topic] = topic_hit_rate

        # Compute accuracies based on topic per question type
        topic_question_type_accuracies = {}
        for q_type in data_main["question_type"].unique():
            topic_question_type_accuracies[q_type] = {}
            qtype_data = data_main[data_main["question_type"] == q_type]
            topic_question_type_counts = qtype_data.groupby("topic")["original_question_id"].nunique().to_dict()
            topic_question_type_correct = qtype_data[qtype_data["hit"] == 1].groupby("topic")["original_question_id"].nunique().to_dict()
            for topic in topic_question_type_counts:
                count = topic_question_type_counts[topic]
                correct = topic_question_type_correct.get(topic, 0)
                accuracy = correct / count if count > 0 else 0.0
                topic_question_type_accuracies[q_type][topic] = accuracy

        # Compute accuracies based on sub_topic
        sub_topic_accuracies = {}
        for sub_topic in data_main["sub_topic"].unique():
            sub_topic_data = data_main[data_main["sub_topic"] == sub_topic]
            sub_topic_correct = sub_topic_data[sub_topic_data["hit"] == 1]["original_question_id"].nunique()
            sub_topic_total = sub_topic_data["original_question_id"].nunique()
            sub_topic_hit_rate = sub_topic_correct / sub_topic_total
            sub_topic_accuracies[sub_topic] = sub_topic_hit_rate

        # Compute accuracies based on sub_topic per question type
        sub_topic_question_type_accuracies = {}
        for q_type in data_main["question_type"].unique():
            sub_topic_question_type_accuracies[q_type] = {}
            qtype_data = data_main[data_main["question_type"] == q_type]
            sub_topic_question_type_counts = qtype_data.groupby("sub_topic")["original_question_id"].nunique().to_dict()
            sub_topic_question_type_correct = qtype_data[qtype_data["hit"] == 1].groupby("sub_topic")["original_question_id"].nunique().to_dict()
            for sub_topic in sub_topic_question_type_counts:
                count = sub_topic_question_type_counts[sub_topic]
                correct = sub_topic_question_type_correct.get(sub_topic, 0)
                accuracy = correct / count if count > 0 else 0.0
                sub_topic_question_type_accuracies[q_type][sub_topic] = accuracy

        # For Recognition questions, compute is_ocr accuracies
        is_ocr_accuracies = {}
        recognition_data = data_main[data_main["question_type"] == "Recognition"]
        if not recognition_data.empty:
            for is_ocr_value in recognition_data["is_ocr"].unique():
                is_ocr_data = recognition_data[recognition_data["is_ocr"] == is_ocr_value]
                is_ocr_correct = is_ocr_data[is_ocr_data["hit"] == 1]["original_question_id"].nunique()
                is_ocr_total = is_ocr_data["original_question_id"].nunique()
                is_ocr_hit_rate = is_ocr_correct / is_ocr_total
                is_ocr_accuracies[is_ocr_value] = is_ocr_hit_rate

        # For Reasoning questions, compute difficulty_level accuracies
        difficulty_level_accuracies = {}
        reasoning_data = data_main[data_main["question_type"] == "Reasoning"]
        if not reasoning_data.empty:
            for level in reasoning_data["difficulty_level"].unique():
                level_data = reasoning_data[reasoning_data["difficulty_level"] == level]
                level_correct = level_data[level_data["hit"] == 1]["original_question_id"].nunique()
                level_total = level_data["original_question_id"].nunique()
                level_hit_rate = level_correct / level_total
                difficulty_level_accuracies[level] = level_hit_rate

        # Prepare the evaluation results dictionary
        eval_results = {
            "overall_accuracy": overall_hit_rate,
            "question_type_accuracies": question_type_accuracies,
            "topic_accuracies": topic_accuracies,
            "topic_question_type_accuracies": topic_question_type_accuracies,
            "sub_topic_accuracies": sub_topic_accuracies,
            "sub_topic_question_type_accuracies": sub_topic_question_type_accuracies,
            "is_ocr_accuracies": is_ocr_accuracies,
            "difficulty_level_accuracies": difficulty_level_accuracies,
            # Include data_main for further processing in utils.py if needed
            "data_main": data_main,
        }

        return eval_results
