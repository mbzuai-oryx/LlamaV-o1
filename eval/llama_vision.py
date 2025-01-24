import torch
from PIL import Image
import os.path as osp
import sys
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import re
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import uuid
import copy
import json
    
class llama_vision(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    # This function is used to split Llama-3.2-90B
    def split_model(self):
        import math
        device_map = {}
        num_gpus = torch.cuda.device_count()
        rank, world_size = get_rank_and_world_size()
        num_gpus = num_gpus // world_size

        num_layers = 100
        # GPU0: -5, GPU-1: -7
        total_cost = num_layers + 5 + 7

        # Since the first GPU will be used for ViT, treat it as 0.8 GPU.
        num_layers_per_gpu = total_cost // num_gpus
        num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
        num_layers_per_gpu[0] -= 5
        num_layers_per_gpu[-1] -= 7

        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
                layer_cnt += 1

        device_map['vision_model'] = rank
        device_map['language_model.model.embed_tokens'] = rank
        device_map['language_model.model.rotary_emb'] = rank
        device_map['language_model.model.norm'] = rank + world_size * (num_gpus - 1)
        device_map['language_model.lm_head'] = rank + world_size * (num_gpus - 1)
        device_map['multi_modal_projector'] = rank + world_size * (num_gpus - 1)
        return device_map

    def __init__(self, model_path='meta-llama/Llama-3.2-11B-Vision-Instruct', **kwargs):
        try:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
        except Exception as e:
            logging.critical('Please install transformers>=4.45.0 before using llama_vision.')
            raise e

        if '90b' in model_path.lower():
            device_map = self.split_model()
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            ).eval()
        else:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map='cpu',
            ).cuda().eval()

        self.device = 'cuda'
        self.processor = AutoProcessor.from_pretrained(model_path)
        if 'Instruct' in model_path:
            kwargs_default = dict(do_sample=True, temperature=0.6, top_p=0.9)
        else:
            kwargs_default = dict(do_sample=False, max_new_tokens=2048, temperature=0.0, top_p=None, num_beams=5)
        kwargs_default = dict(do_sample=True, max_new_tokens=2048, temperature=0.6, top_p=0.9)
        kwargs.update(kwargs_default)
        print(f'Following kwargs received: {kwargs}, will use as generation config. ')
        self.kwargs = kwargs
        self.model_name = model_path

    
    def judge(self, image, prompt, outputs, type="summary", caption=None, reasoning=None):
        input_outputs = outputs
        
        hint = None
        if type == "summary":
            judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide better provides a summary of what it should do to solve the question. The summary should focus on outlining the main approach instead of stating specific analytical reasoning or math formula.'
            recall_prompt = f'Please note that a better summary should focus on outlining the main approach instead of stating specific analytical reasoning or math formula.'
        elif type == "caption":
            judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide better summarizes the information in the image related to the question, and has fewer errors. It is essential that the captions are as thorough as possible while remaining accurate, capturing as many details as possible rather than providing only general commentary.'
            recall_prompt = f'Please note that a better caption should be as thorough as possible while remaining accurate, capturing as many details as possible rather than providing only general commentary.'
        elif type == "reasoning":
            judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide better explains the reasoning process to solve the question, and has fewer errors. Begin by thoroughly reviewing the question, followed by an in-depth examination of each answer individually, noting any differences. Subsequently, analyze these differences to determine which response demonstrates stronger reasoning and provide a clear conclusion.'
            recall_prompt = f'Begin by thoroughly reviewing the question, followed by an in-depth examination of each answer individually, noting any differences. Subsequently, analyze these differences to determine which response demonstrates stronger reasoning and provide a clear conclusion.'
            hint = caption
        elif type == "conclusion":
            judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide offers a more effective conclusion to the question. The conclusion should align with the reasoning presented in the hint. The conclusion should never refuse to answer the question.'
            recall_prompt = f'Please note that a better conclusion should align with the reasoning presented in the hint. The conclusion should never refuse to answer the question.'
            hint = caption + " " + reasoning
        if type == "reasoning":
            reasoning_prompt = f"""Now you act as a judge, helping me determine whether the reasoning process in the given text is correct and accurate based on the given information.
            You should assume that the given information about the image is correct.
            You should only consider the reasoning process itself, not the correctness of the background information.  
            If the reasoning process involves any calculations, you should verify the accuracy of the calculations.
            You should output 'correct' if you don't find any errors in the reasoning process, and 'incorrect' if you find any errors."""
            
            reasoning_prompt_1 = reasoning_prompt + f'\n\nGiven Information: {hint}' + f'\n\nReasoning Process: {input_outputs[0]}'
            reasoning_message_1 = [
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': reasoning_prompt_1}
                ]}
            ]
            reasoning_input_text_1 = self.processor.apply_chat_template(reasoning_message_1, add_generation_prompt=True)
            reasoning_inputs_1 = self.processor(None, reasoning_input_text_1, return_tensors='pt').to(self.device)
            reasoning_output_1 = self.model.generate(**reasoning_inputs_1, **self.kwargs)
            reasoning_output_text_1 = self.processor.decode(reasoning_output_1[0][reasoning_inputs_1['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace('<|endoftext|>', '')
            if "incorrect" in reasoning_output_text_1:
                #logging
                with open('log.jsonl', 'a') as f:
                    json_obj = {
                        "prompt": prompt,
                        "outputs": outputs,
                        "judge_output": reasoning_output_text_1
                    }
                    f.write(json.dumps(json_obj) + '\n')
                return 1
            
            reasoning_prompt_2 = reasoning_prompt + f'\n\nGiven Information: {hint}' + f'\n\nReasoning Process: {input_outputs[1]}'
            reasoning_message_2 = [
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': reasoning_prompt_2}
                ]}
            ]
            reasoning_input_text_2 = self.processor.apply_chat_template(reasoning_message_2, add_generation_prompt=True)
            reasoning_inputs_2 = self.processor(None, reasoning_input_text_2, return_tensors='pt').to(self.device)
            reasoning_output_2 = self.model.generate(**reasoning_inputs_2, **self.kwargs)
            reasoning_output_text_2 = self.processor.decode(reasoning_output_2[0][reasoning_inputs_2['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace('<|endoftext|>', '')
            if "incorrect" in reasoning_output_text_2:
                #logging
                with open('log.jsonl', 'a') as f:
                    json_obj = {
                        "prompt": prompt,
                        "outputs": outputs,
                        "judge_output": reasoning_output_text_2
                    }
                    f.write(json.dumps(json_obj) + '\n')
                return 0
                
        judge_prompt += f'\n\nQuestion: {prompt}'
        if hint:
            judge_prompt += f'\n\nHint about the Question: {hint}'
        for i, output in enumerate(input_outputs):
            judge_prompt += f'\nRepsonse {i+1}: {output}'
        judge_prompt += f'\n\n{recall_prompt}'
        judge_prompt += f' Please strictly follow the following format requirements when outputting, and don’t have any other unnecessary words.'
        judge_prompt += f'\n\nOutput format: "Since [reason], I choose response [1/2]."'
        
        judge_message = [
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': judge_prompt}
            ]}
        ]
        judge_input_text = self.processor.apply_chat_template(judge_message, add_generation_prompt=True)
        judge_inputs = self.processor(image, judge_input_text, return_tensors='pt').to(self.device)
        judge_output = self.model.generate(**judge_inputs, **self.kwargs)
        judge_output_text = self.processor.decode(judge_output[0][judge_inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace('<|endoftext|>', '')
        
        # log to log.jsonl (json format){"prompt": prompt, "outputs": outputs, "judge_output": judge_output_text}
        with open('log.jsonl', 'a') as f:
            json_obj = {
                "prompt": prompt,
                "outputs": outputs,
                "judge_output": judge_output_text
            }
            f.write(json.dumps(json_obj) + '\n')
        
        if "I choose response 1" in judge_output_text:
            return 0
        else:
            return 1
    def use_custom_prompt(self, dataset):
        if dataset is None:
            return False
        if listinstr(['AI2D', 'MMMU', 'MathVista', 'ChartQA', 'DocVQA'], dataset):
            # For Certain dataset we use custom prompt
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        if listinstr(['AI2D'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            for key, item in options.items():
                question += f'\n{key}. {item}'
            if '11B' in self.model_name:
                prompt = (
                    f'Look at the scientific diagram carefully and answer the following question: {question}\n'
                    f'Respond only with the correct option digit.'
                )
            elif '90B' in self.model_name:
                prompt = (
                    f'Look at the scientific diagram carefully and answer the following question: {question}\n'
                    f'Respond only with the correct option digit.'
                )
        elif listinstr(['MMMU'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            options = '\n'.join([f'{key}. {item}' for key, item in options.items()])
            prompt = (
                f'Look at the image carefully and solve the following question step-by-step. '
                f'Question: {question} Options: {options} Indicate the correct answer at the end.'
            )
            for i in range(len(tgt_path)):
                prompt = prompt.replace(f'<image {i+1}>', '')
        elif listinstr(['MathVista'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            prompt = f'{question}'
        elif listinstr(['ChartQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            if '11B' in self.model_name:
                prompt = (
                    f'You are provided a chart image and will be asked a question. '
                    f'You have to think through your answer and provide a step-by-step solution. '
                    f'Once you have the solution, write the final answer in at most a few words at the end '
                    f"with the phrase \"FINAL ANSWER:\". "
                    f"The question is: {question}<cot_start>Let's think step by step."
                )
            elif '90B' in self.model_name:
                prompt = (
                    f'You are provided a chart image and will be asked a question. '
                    f'Follow these steps carefully:\n '
                    f'Step 1: Analyze the question to understand what specific data or information is being asked for. '
                    f'Focus on whether the question is asking for a specific number or category '
                    f'from the chart image.\n '
                    f'Step 2: Identify any numbers, categories, or groups mentioned in the question '
                    f'and take note of them. Focus on detecting and matching them directly to the image. \n'
                    f'Step 3: Study the image carefully and find the relevant data corresponding to the categories '
                    f'or numbers mentioned. Avoid unnecessary assumptions or calculations; '
                    f'simply read the correct data from the image.\n '
                    f'Step 4: Develop a clear plan to solve the question by locating the right data. '
                    f'Focus only on the specific category or group that matches the question. \n'
                    f'Step 5: Use step-by-step reasoning to ensure you are referencing the correct numbers '
                    f'or data points from the image, avoiding unnecessary extra steps or interpretations.\n '
                    f"Step 6: Provide the final answer, starting with \"FINAL ANSWER:\" "
                    f'and using as few words as possible, '
                    f'simply stating the number or data point requested. \n\n '
                    f"The question is: {question}<cot_start>Let's think step by step."
                )
        elif listinstr(['DocVQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            prompt = (
                f'Read the text in the image carefully and answer the question '
                f'with the text as seen exactly in the image. '
                f'For yes/no questions, just respond Yes or No. '
                f'If the answer is numeric, just respond with the number and nothing else. '
                f'If the answer has multiple words, just respond with the words and absolutely nothing else. '
                f'Never respond in a sentence or a phrase.\n Question: {question}'
            )
        else:
            raise NotImplementedError(f'Dataset {dataset}) not supported.')

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message
        
    
    
    def generate_inner_stage_beam(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path)
        main_prompts = {
            "summary": "\nSummarize how you will approach the problem and explain the steps you will take to reach the answer.",
            "caption": "Provide a detailed description of the image, particularly emphasizing the aspects related to the question.",
            "reasoning": "Provide a chain-of-thought, logical explanation of the problem. This should outline step-by-step reasoning.",
            "conclusion": "State the final answer in a clear and direct format. It must match the correct answer exactly.",
        }
        prompt_names = ["summary", "caption", "reasoning", "conclusion"]
        messages = [
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': prompt+main_prompts['summary']}
            ]}
        ]
        
        reasoning = None
        results = {}
        self.kwargs['max_new_tokens'] = 1024
        self.kwargs['do_sample'] = False
        for i, prompt_name in enumerate(prompt_names):
            prompt_val = main_prompts[prompt_name]
            candidates = []
            for _ in range(4):  
                input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(image, input_text, return_tensors='pt').to(self.device)
                output = self.model.generate(**inputs, **self.kwargs)
                
                new_generated_ids = output[0]
                generated_text = self.processor.tokenizer.decode(new_generated_ids[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                candidates.append({
                    'input_ids': new_generated_ids.unsqueeze(0),
                    'generated_text': generated_text,
                })
            while(len(candidates) > 1):
                # randomly select two candidates
                candidate1 = candidates.pop(np.random.randint(len(candidates)))
                candidate2 = candidates.pop(np.random.randint(len(candidates)))
                outputs = [candidate1['generated_text'], candidate2['generated_text']]
                try:
                    caption_val = results.get("caption", None)
                    reasoning_val = results.get("reasoning", None)
                    best_index = self.judge(image, prompt, outputs, type=prompt_name, caption=caption_val, reasoning=reasoning_val)
                except Exception as e:
                    print("[WARNING] Skipping a judge due to", str(e))
                    best_index = 0
                if best_index == 0:
                    candidates.append(candidate1)
                else:
                    candidates.append(candidate2)
            results[prompt_name] = candidates[0]['generated_text']
            messages.extend([
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": candidates[0]['generated_text']}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "" if i + 1 >= len(prompt_names) else prompt_names[i+1]}]
                }
            ])
        reasoning = results['reasoning']
        final_output = candidates[0]['generated_text']
        return final_output
    
    def generate_inner(self, message, dataset=None):
        return self.generate_inner_stage_beam(message, dataset)


