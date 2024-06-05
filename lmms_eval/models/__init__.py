import os

AVAILABLE_MODELS = {
    "llava": "Llava",
    "llava_hf": "LlavaHf",
    "llava_sglang": "LlavaSglang",
    "qwen_vl": "Qwen_VL",
    "fuyu": "Fuyu",
    "gpt4v": "GPT4V",
    "instructblip": "InstructBLIP",
    "minicpm_v": "MiniCPM_V",
    "mblip":"mBLIP",
    "llava_vcd": "Llava_vcd",
    "idefics2": "Idefics2",
    "qwen_vl_api": "Qwen_VL_API",
    "phi3v": "Phi3v",
}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except ImportError as e:
        print(f"Failed to import {model_class}: {e}")


import hf_transfer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
