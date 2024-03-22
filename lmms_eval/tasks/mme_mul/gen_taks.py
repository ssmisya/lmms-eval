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

ref_file_path="/mnt/petrelfs/songmingyang/code/tools/lmms-eval/lmms_eval/tasks/mme_mul_tasks/mme_mul_ar.yaml"
output_path="/mnt/petrelfs/songmingyang/code/tools/lmms-eval/lmms_eval/tasks/mme_mul_tasks"
languages = "en ar  bg  bn  de  el  es  fa  fr  gu  hi  id  it  ja  jv  ko  ml  mr  ms  my  nl  pt  ru  sv  sw  ta  te  th  tr  uk  ur  vi  zh"
languages = languages.split()
type_list=[
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
        "commonsense_reasoning",
        "numerical_calculation",
        "text_translation",
        "code_reasoning",
    ]
# 打开并读取YAML文件
with open(ref_file_path, 'r') as f:
    ref_data = yaml.safe_load(f)

# with open(group_file_path,'r') as f:
#     group_data = yaml.safe_load(f)

# group_data["task"] = []
    
for language in tqdm(languages):
    data_list = []
    for type_item in type_list:
        data_path = f'/mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/MME/MME_Benchmark_release_version/{type_item}/json_labels/mme_{type_item}_{language}.json'
        data_list.append(data_path)
    ref_data["dataset_kwargs"] = {
        "data_files": {
            "test": data_list
        },
        "token":True
    }
    # ref_data["dataset_kwargs"]["data_files"] =  {'test':data_list}
    ref_data["task"] = f'mme_mul_{language}'
    # ref_data["group"] = 
    with open(f'{output_path}/mme_mul_{language}.yaml', 'w') as f:
        yaml.safe_dump(ref_data, f)

# with open(group_file_path, 'w') as f:
#     yaml.safe_dump(group_data, f)

