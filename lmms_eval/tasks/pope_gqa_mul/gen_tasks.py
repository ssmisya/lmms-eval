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
task_name="pope_gqa_mul"
ref_file_path=f"/mnt/petrelfs/songmingyang/code/tools/lmms-eval/lmms_eval/tasks/{task_name}_tasks/{task_name}_ar.yaml"
output_path=f"/mnt/petrelfs/songmingyang/code/tools/lmms-eval/lmms_eval/tasks/{task_name}_tasks"
languages = "en ar  bg  bn  de  el  es  fa  fr  gu  hi  id  it  ja  jv  ko  ml  mr  ms  my  nl  pt  ru  sv  sw  ta  te  th  tr  uk  ur  vi  zh"
languages = languages.split()
# 打开并读取YAML文件
with open(ref_file_path, 'r') as f:
    ref_data = yaml.safe_load(f)

for language in tqdm(languages):
    ref_data["dataset_kwargs"]["data_files"] =  {'test':f'/mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/pope_mul/gqa/{language}/gqa_pope_popular_{language}.json'}
    ref_data["task"] = f'{task_name}_{language}'
    ref_data["group"] = task_name
    with open(f'{output_path}/{task_name}_{language}.yaml', 'w') as f:
        yaml.safe_dump(ref_data, f)

