# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import logging
import requests
import torch
import string
import random
from typing import Dict, List, Optional
from vllm import LLM, SamplingParams, EngineArgs
from types import SimpleNamespace

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from quest_pytorch.quest_patch import quest_patch_vllm

class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}

        self.pipeline = None
        self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            output = self.pipeline(text_inputs=prompts, **self.generation_kwargs, )
            assert len(output) == len(prompts)
            # output in the form of a list of list of dictionaries
            # outer list len = batch size
            # inner list len = 1
            generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results

class VLLM_Model:

    def __init__(self, name_or_path: str, **generation_kwargs) -> None:

        self.model = LLM(model=name_or_path, dtype=torch.bfloat16, gpu_memory_utilization=0.8, trust_remote_code=True)

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_single_input([prompt], **kwargs)

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
    
        outputs_ = self.model.generate(prompts, use_tqdm=False)
        generated_texts = []
        text = outputs_[0].outputs[0].text
        generated_texts.append({'text': [text]})

        return generated_texts

    def process_single_input(self, prompts: List[str], **kwargs):
        # sampling_params = SamplingParams(temperature=0.8, top_p=0.95, top_k=32)
        outputs_ = self.model.generate(prompts, use_tqdm=False)
        text = outputs_[0].outputs[0].text

        return text

class VLLM_quest_Model:

    def __init__(self, name_or_path: str, **generation_kwargs) -> None:

        # customed_cache_config = config.CacheConfig(block_size=64, gpu_memory_utilization=0.9, swap_space=4.0, cache_dtype='auto')
        args = generation_kwargs.get('args', None)
        model = LLM(model=name_or_path, dtype=torch.bfloat16, gpu_memory_utilization=0.8, trust_remote_code=True, enforce_eager=True, block_size = 16)
        self.model = quest_patch_vllm(model, args)
        self.args = args

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_single_input([prompt], **kwargs)

    def process_single_input(self, prompts: List[str], **kwargs):

        sampling_params = SamplingParams(temperature=self.args.temperature, top_p=self.args.top_p, top_k=self.args.top_k, max_tokens=self.args.max_tokens)
        outputs_ = self.model.generate(prompts, use_tqdm=False, sampling_params=sampling_params)
        text = outputs_[0].outputs[0].text
        
        return text


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = torch.device('cuda')

    args = {"token_budget":4096, "quest":True, "temperature":1.0, "top_p":1.0, "top_k":32, "max_tokens":262000}
    args = SimpleNamespace(**args)

    model_path = "/remote-home/pengyichen/RULER_pyc/scripts/model/llama3-8b-instruct-262k"
    model = VLLM_quest_Model(model_path, args=args)
    
    file_path = '/remote-home/pengyichen/RULER_pyc/scripts/pred/long_text.txt'
    output_path = '/remote-home/pengyichen/RULER_pyc/scripts/pred/long_output.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        loaded_text = file.read()
    
    output = model.process_single_input(loaded_text)
    print(output)