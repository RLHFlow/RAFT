import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from datasets import load_dataset



tqdm.pandas()

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="uf_split0_responses_K8.jsonl",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="uf_split0_responses_K8_reward.json",
        metadata={"help": "the location of the output file"},
    )
    

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

ds_dir = script_args.dataset_name_or_path
ds = load_dataset("json", data_files=ds_dir, split="train")

def modify_sample(example):
    idx = np.argmax(example['rewards'])
    example["messages"] = example['prompt'] + [{"role":"user", "content":example['responses'][idx] }]
  
    return example

ds2 = ds.map(modify_sample)
ds3 = ds.remove_columns(["prompt", "responses", "rewards"])
ds3.push_to_hub(script_args.output_dir)

'''
with open(script_args.output_dir, "w", encoding="utf8") as f:
    for i in range(len(gathered_data)):
        json.dump(gathered_data[i], f, ensure_ascii=False)
        f.write('\n')
'''
