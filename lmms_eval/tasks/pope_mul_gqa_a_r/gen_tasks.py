import yaml
from tqdm import tqdm

def function_constructor(loader, node):
    # 定义如何处理`!function`标签
    value = loader.construct_scalar(node)
    return value

yaml.SafeLoader.add_constructor('!function', function_constructor)


def function_representer(dumper, data):
    if isinstance(data,str) and data.startswith("utils"):
        return dumper.represent_scalar('!function', data)
    else:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

# 将表示器添加到表示器字典中
yaml.SafeDumper.add_representer(str, function_representer)

dataset="coco"
ref_file_path=f"/mnt/petrelfs/songmingyang/code/tools/lmms-eval/lmms_eval/tasks/pope_mul_{dataset}_a_r_tasks/pope_mul_{dataset}_a_r_zh.yaml"
group_file_path=f"/mnt/petrelfs/songmingyang/code/tools/lmms-eval/lmms_eval/tasks/pope_mul_{dataset}_a_r/pope_mul_{dataset}_a_r.yaml"
output_path=f"/mnt/petrelfs/songmingyang/code/tools/lmms-eval/lmms_eval/tasks/pope_mul_{dataset}_a_r_tasks"

languages = "en ar  bg  bn  de  el  es  fa  fr  gu  hi  id  it  ja  jv  ko  ml  mr  ms  my  nl  pt  ru  sv  sw  ta  te  th  tr  uk  ur  vi  zh"
# languages = "en ar zh"
languages = languages.split()
# 打开并读取YAML文件
with open(ref_file_path, 'r') as f:
    ref_data = yaml.safe_load(f)


group_data={
    "group": f"pope_mul_{dataset}_a_r",
    "metadata":{"version":0.0},
    "task": []
}
    
for language in tqdm(languages):
    ref_data_files = [f"/mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/pope_mul_reconstructed/{dataset}/{language}/{dataset}_pope_{style}_{language}.json" for style in ["adversarial","random"]]
    ref_data["dataset_kwargs"]["data_files"] =  {'test':ref_data_files}
    ref_data["task"] = f'pope_mul_{dataset}_a_r_{language}'
    ref_data["group"] = f'pope_mul_{dataset}_a_r'
    group_data["task"].append(f"pope_mul_{dataset}_a_r_{language}")
    with open(f'{output_path}/pope_mul_{dataset}_a_r_{language}.yaml', 'w') as f:
        yaml.safe_dump(ref_data, f)

with open(group_file_path, 'w') as f:
    yaml.safe_dump(group_data, f)

