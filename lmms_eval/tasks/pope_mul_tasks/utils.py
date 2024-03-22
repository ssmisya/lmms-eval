# Add the following functions to your existing utils.py file
import os
from PIL import Image
from m3apo.vcd.experiments.eval.language_dict import language_dict

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

def pope_doc_to_visual(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    if os.path.isabs(doc["image"]):
        image_file = doc["image"]
    else:
        if doc["image"].startswith("COCO"):
            image_folder = "/mnt/petrelfs/share_data/quxiaoye/VCD_file/val2014"
        else:
            raise ValueError(f"Unknown image source: {doc['image']}")
        image_file = os.path.join(image_folder, doc["image"])
    img = [Image.open(image_file).convert('RGB')]
    return img


def pope_doc_to_text(doc):
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["text"].strip()
    language = doc["language"].strip()
    yes_word = language_dict[language]['yes'] if isinstance(language_dict[language]['yes'], str) else language_dict[language]['yes'][0]
    no_word = language_dict[language]['no'] if isinstance(language_dict[language]['no'], str) else language_dict[language]['no'][0]
    prompt_suffix = language_dict[language]['prompt_suffix'].format(yes_word,no_word)
    full_prompt = question+" "+prompt_suffix
    return full_prompt


def pope_process_results(doc, results):
    pred = results[0].lower().strip()
    gt_ans = doc["label"].lower().strip()
    language = doc["language"].strip()
    assert gt_ans in ["yes", "no"]
    
    if compare_str_list(language_dict[language]['yes'],pred) and compare_str_list(language_dict[language]['no'],pred):
        score = 0.0
    else:
        score = 1.0 if compare_str_list(language_dict[language][gt_ans],pred) else 0.0
    
    return {
        "pope_accuracy": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans, "language": language},
        "pope_precision": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans, "language": language},
        "pope_recall": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans, "language": language},
        "pope_f1_score": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans, "language": language},
        "pope_yes_ratio": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans, "language": language},
    }


def pope_aggregate_accuracy(results):
    total_score = 0
    for result in results:
        total_score += result["score"]
    avg_score = total_score / len(results)
    return avg_score


def pope_aggregate_precision(results):
    true_positives = 0
    false_positives = 0
    for result in results:
        pred = result["prediction"]
        gt = result["ground_truth"]
        language = result["language"]
        pos = language_dict[language]['yes']
        if compare_str_list(language_dict[language]['yes'],pred) and compare_str_list(language_dict[language]['no'],pred):
            continue
        if gt == "yes" and compare_str_list(pos,pred):
            true_positives += 1
        elif gt == "no" and compare_str_list(pos,pred):
            false_positives += 1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision


def pope_aggregate_recall(results):
    true_positives = 0
    false_negatives = 0
    for result in results:
        pred = result["prediction"]
        gt = result["ground_truth"]
        language = result["language"]
        pos = language_dict[language]['yes']
        neg = language_dict[language]['no']
        if compare_str_list(language_dict[language]['yes'],pred) and compare_str_list(language_dict[language]['no'],pred):
            continue
        if gt == "yes" and compare_str_list(pos,pred):
            true_positives += 1
        elif gt == "yes" and compare_str_list(neg,pred):
            false_negatives += 1
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall


def pope_aggregate_f1_score(results):
    precision = pope_aggregate_precision(results)
    recall = pope_aggregate_recall(results)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def pope_aggregate_yes_ratio(results):
    yes_count = 0
    no_count = 0
    for result in results:
        gt = result["ground_truth"]
        if gt == "yes":
            yes_count += 1
        elif gt == "no":
            no_count += 1
    yes_ratio = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    return yes_ratio
