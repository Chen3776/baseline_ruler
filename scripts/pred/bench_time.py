import os
import json
import logging
import requests
import torch
import string
import random
import time
from typing import Dict, List, Optional
from vllm import LLM, SamplingParams, EngineArgs
from types import SimpleNamespace

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from quest_pytorch.quest_patch import quest_patch_vllm
from model_wrappers import VLLM_Model, VLLM_quest_Model


os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device('cuda')
args = {"token_budget":4096, "quest":True, "temperature":1.0, "top_p":1.0, "top_k":32, "max_tokens":262000}
args = SimpleNamespace(**args)

model_path = "/remote-home/pengyichen/RULER_pyc/scripts/model/llama3-8b-instruct-262k"

model = VLLM_Model(model_path, args=args, block_size = 32)

file_path = '/remote-home/pengyichen/RULER_pyc/scripts/pred/long_text.txt'
output_path = '/remote-home/pengyichen/RULER_pyc/scripts/pred/long_output.txt'

with open(file_path, 'r', encoding='utf-8') as file:
    loaded_text = file.read()

t1 = time.time()
output = model.process_single_input(loaded_text)
t2 = time.time()

elapsed_time = t2 - t1
print(f"函数运行时间: {elapsed_time:.2f}秒")