from collections import defaultdict
import os
import datetime
import json
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from PIL import Image
from m3apo.vcd.experiments.eval.language_dict import language_dict

import logging

eval_logger = logging.getLogger("lmms-eval")

dir_name = os.path.dirname(os.path.abspath(__file__))

eval_type_dict = {
    "Perception": [
        "existence",
        "count",
        "position",
        "color",
        "posters",
        "celebrity",
        "scene",
        "landmark",
        "artwork",
        "OCR",
    ],
    "Cognition": [
        "commonsense_reasoning",
        "numerical_calculation",
        "text_translation",
        "code_reasoning",
    ],
}


# replace_prompt = " Please answer yes or no."


def compare_str_list(str1,str2):
    if isinstance(str1,list):
        for i in str1:
            if i in str2:
                return True
        return False
    elif isinstance(str1,str):
        return str1 in str2
    else:
        raise ValueError("str1 should be list or str")


def mme_doc_to_visual(doc):
    if os.path.isabs(doc["image"]):
        image_file = doc["image"]
    else:
        image_folder = "/mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/MME/MME_Benchmark_release_version"
        image_file = os.path.join(image_folder, doc["image"])
    img = [Image.open(image_file).convert('RGB')]
    return img




def mme_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    pred_ans = results[0].lower().strip()
    gt_ans = doc["label"].lower().strip().replace(".", "")
    language = doc["language"].strip()
    assert gt_ans in ["yes", "no"]
    score = 1.0 if compare_str_list(language_dict[language][gt_ans],pred_ans) else 0.0
    category = doc["category"]
    key_name = "mme_percetion_score" if category in eval_type_dict["Perception"] else "mme_cognition_score"
    category_acc = f"mme_{category}_acc"
    category_acc_plus = f"mme_{category}_acc_plus"
    
    # Note: the key name here is very important. It decides which aggregation function will receive the results
    # We note down the question id/category to help us aggregate the results later
    return{
        key_name: {"question_id": doc["question_id"], "category": category, "score": score},
        category_acc: {"question_id": doc["question_id"], "category": category, "score": score},
        category_acc_plus: {"question_id": doc["question_id"], "category": category, "score": score},
    }





def mme_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = defaultdict(dict)
    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        category = result["category"]
        if question_id not in category2score[category]:
            category2score[category][question_id] = []
        category2score[category][question_id].append(score)
    category2avg_score = {}
    for category, question2scores in category2score.items():
        total_score = 0
        for question_id, scores in question2scores.items():
            assert len(scores) == 2 , f"question_id: {question_id}, scores: {scores}"
            acc = sum(scores) / len(scores) * 100.0
            acc_plus = (sum(scores) == 2) * 100.0
            score = acc_plus + acc
            total_score += score
        avg_score = total_score / len(question2scores)
        category2avg_score[category] = avg_score
    for category, avg_score in category2avg_score.items():
        eval_logger.info(f"{category}: {avg_score:.2f}")
    total_score = sum(category2avg_score.values())
    return total_score

def mme_category_acc(results):
    acc_list = [acc["score"] for acc in results]
    return sum(acc_list) / len(acc_list) * 100.0

def mme_category_acc_plus(results):
    question2score = {}
    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        if question_id not in question2score:
            question2score[question_id] = []
        question2score[question_id].append(score)
    acc_plus_list = []
    for question_id, scores in question2score.items():
        assert len(scores) == 2
        acc_plus = (sum(scores) == 2) * 100.0
        acc_plus_list.append(acc_plus)
    return sum(acc_plus_list) / len(acc_plus_list)
        
