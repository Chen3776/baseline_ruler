import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import math
import time
from tqdm import tqdm
from pathlib import Path
import traceback
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from vllm import LLM, SamplingParams
import torch.distributed as dist

SERVER_TYPES = (
    'vllm',
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values

parser = argparse.ArgumentParser()
# Data
parser.add_argument("--data_dir", default='/remote-home/pengyichen/RULER_pyc/result_pyc/llama2-7b-chat/synthetic/131072/data', help='path to load the dataset jsonl files')
parser.add_argument("--save_dir", default='/remote-home/pengyichen/RULER_pyc/result_pyc/llama2-7b-chat/synthetic/131072/data', help='path to save the prediction jsonl files')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--task", type=str, default = 'niah_single_1', help='Options: tasks in benchmark')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')

# Server
parser.add_argument("--server_type", default='vllm', action=ServerAction, choices=SERVER_TYPES)
parser.add_argument("--server_host", type=str, default='127.0.0.1')
parser.add_argument("--server_port", type=str, default='5000')
parser.add_argument("--ssh_server", type=str)
parser.add_argument("--ssh_key_path", type=str)
# parser.add_argument("--model_name_or_path", type=str, default='/remote-home/pengyichen/RULER_pyc/scripts/model/chatglm3-6b-128k')
parser.add_argument("--model_name_or_path", type=str, default='/remote-home/pengyichen/RULER_pyc/scripts/model/llama3-8b-instruct-262k')

# Inference
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default='')
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)

args = parser.parse_args()

args.stop_words = list(filter(None, args.stop_words.split(',')))

if args.server_type == 'hf':
    args.threads = 1


def get_llm(tokens_to_generate):
    if args.server_type == 'vllm':
        from model_wrappers import VLLM_Model
        llm = VLLM_Model(
            name_or_path=args.model_name_or_path,
            do_sample=args.temperature > 0,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )
    return llm
    
def main():

    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")


    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')
        
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    task_file = args.data_dir + '/' + args.task + '/' + f'{args.subset}.jsonl'
    
    if args.chunk_amount > 1:
        pred_file = args.save_dir +'/'+ f'{args.task}-{args.chunk_idx}.jsonl'
    else:
        pred_file = args.save_dir +'/'+ f'{args.task}.jsonl'
        
    print(f'Predict {args.task} \nfrom {task_file}\nto {pred_file}')
    # pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file):
        pred_index = [sample['index'] for sample in read_manifest(pred_file)]
        data = [sample for sample in read_manifest(task_file) if sample['index'] not in pred_index]
    else:
        data = read_manifest(task_file)

    # Load api
    llm = get_llm(config['tokens_to_generate'])

    def process_serial(llm, idx, index, input_, outputs_list, others, truncation, length):
        
        pred_text = llm.process_single_input(prompts=[input_])
        outputs_serial.append(
                {
                    'index': index,
                    'pred': pred_text,
                    'input': input_,
                    'outputs': outputs_list[0],
                    'others': others,
                    'truncation': truncation,
                    'length': length,
                }
            )
    outputs_serial = []

    batched_data = []
    batch = []
    
    for idx, data_point in enumerate(data):
        data_point['idx'] = idx

        if len(batch) >= args.batch_size:
            batched_data.append(batch)
            batch = []

        batch.append(data_point)

    if len(batch):
        batched_data.append(batch)

    # 锁定输入batch = 1
    idx = 0
    with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        for batch in tqdm(batched_data):
            data_point = batch[0]
        
            process_serial(
                llm,
                idx,
                data_point['index'],
                data_point['input'],
                data_point['outputs'],
                data_point.get('others', {}),
                data_point.get('truncation', -1),
                data_point.get('length', -1),
            )

            output = outputs_serial[idx]
            fout.write(json.dumps(output) + '\n')
            idx += 1

if __name__ == '__main__':
    main()
