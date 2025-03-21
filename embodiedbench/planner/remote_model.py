import json
import sys
import os
import base64
import anthropic
import google.generativeai as genai
from openai import OpenAI,AzureOpenAI
import typing_extensions as typing
import lmdeploy
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from embodiedbench.planner.planner_config.generation_guide_manip import llm_generation_guide_manip, vlm_generation_guide_manip
from embodiedbench.planner.planner_utils import convert_format_2claude, convert_format_2gemini, ActionPlan_1, ActionPlan, ActionPlan_lang, ActionPlan_1_manip, ActionPlan_manip, ActionPlan_lang_manip

temperature = 0
max_completion_tokens = 2048
remote_url = os.environ.get('remote_url')

class RemoteModel:
    def __init__(
        self,
        model_name,
        model_type='remote',
        language_only=False,
        tp=1,
        task_type=None # used to distinguish between manipulation and other environments
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.language_only = language_only
        self.task_type = task_type

        self.api_cost = 0


        if self.model_type == 'local':
            backend_config = PytorchEngineConfig(session_len=12000, dtype='float16', tp=tp)
            self.model = pipeline(self.model_name, backend_config=backend_config)
        else:
            # if "claude" in self.model_name:
            #     self.model = anthropic.Anthropic(
            #         api_key=os.environ.get("ANTHROPIC_API_KEY"),
            #     )
            # elif "gemini" in self.model_name:
            #     self.model = OpenAI(
            #         api_key=os.environ.get("GEMINI_API_KEY"),
            #         base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            #     )
            if self.model_name.startswith("gpt"):
                # self.model = OpenAI()
                self.model = AzureOpenAI(
                            azure_endpoint = "https://responsible-ai.openai.azure.com/", 
                            api_key= os.environ['AZURE_OPENAI_API_KEY'],  
                            api_version= "2024-10-01-preview",
                            )
            # elif 'qwen' in self.model_name:
            #     self.model = OpenAI(
            #         api_key=os.getenv("DASHSCOPE_API_KEY"),
            #         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            #     )
            # elif "Qwen2-VL" in self.model_name:
            #     self.model = OpenAI(base_url = remote_url)
            # elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
            #     self.model = OpenAI(base_url = remote_url)
            # elif "OpenGVLab/InternVL" in self.model_name:
            #     self.model = OpenAI(base_url = remote_url)
            # elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
            #     self.model = OpenAI(base_url = remote_url)
            # elif "90b-vision-instruct" in self.model_name: # you can use fireworks to inference
            #     self.model = OpenAI(base_url='https://api.fireworks.ai/inference/v1',
            #                         api_key=os.environ.get("firework_API_KEY"))
            else:
                from together import Together
                self.model = Together()
                # self.model = OpenAI(base_url = "http://10.225.68.29:1703/v1", api_key="yyy")
                # except:
                #     raise ValueError(f"Unsupported model name: {model_name}")


    def respond(self, message_history: list):
        if self.model_type == 'local':
            return self._call_local(message_history)
        else:
            return self._call_gpt(message_history)
            # if "claude" in self.model_name:
            #     return self._call_claude(message_history)
            # elif "gemini" in self.model_name:
            #     return self._call_gemini(message_history)
            # elif "gpt" in self.model_name:
            #     return self._call_gpt(message_history)
            # elif 'qwen' in self.model_name:
            #     return self._call_gpt(message_history)
            # elif "Qwen2-VL-7B-Instruct" in self.model_name:
            #     return self._call_qwen7b(message_history)
            # elif "Qwen2-VL-72B-Instruct" in self.model_name:
            #     return self._call_qwen72b(message_history)
            # elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
            #     return self._call_llama11b(message_history)
            # elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
            #     return self._call_qwen72b(message_history)
            # elif "90b-vision-instruct" in self.model_name:
            #     return self._call_llama90(message_history)
            # elif "OpenGVLab/InternVL" in self.model_name:
            #     return self._call_intern38b(message_history)
            # # elif "OpenGVLab/InternVL2_5-38B" in self.model_name:
            # #     return self._call_intern38b(message_history)
            # # elif "OpenGVLab/InternVL2_5-78B" in self.model_name:
            # #     return self._call_intern38b(message_history)
            # else:
            #     raise ValueError(f"Unsupported model name: {self.model_name}")

    def _call_local(self, message_history: list):
        if self.task_type == 'manip':
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "embodied_planning",
                    "schema": llm_generation_guide_manip if self.language_only else vlm_generation_guide_manip
                }
            }
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "embodied_planning",
                    "schema": llm_generation_guide if self.language_only else vlm_generation_guide
                }
            }
        response = self.model(
            message_history,
            gen_config=GenerationConfig(
                temperature=temperature,
                response_format=response_format,
                max_new_tokens=max_completion_tokens,
            )
        )
        out = response.text
        return out

    def _call_claude(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2claude(message_history)

        response = self.model.messages.create(
            model=self.model_name,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            messages=message_history
        )

        return response.content[0].text 

    def _call_gemini(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if self.task_type == 'manip':
            response = self.model.beta.chat.completions.parse(
                model=self.model_name, 
                messages=message_history,
                response_format= ActionPlan_lang_manip if self.language_only else ActionPlan_manip,
                temperature=temperature,
                max_tokens=max_completion_tokens
            )
        else:
            response = self.model.beta.chat.completions.parse(
                model=self.model_name, 
                messages=message_history,
                response_format= ActionPlan_lang if self.language_only else ActionPlan,
                temperature=temperature,
                max_tokens=max_completion_tokens
            )
        tokens = response.usage.prompt_tokens

        return str(response.choices[0].message.parsed.model_dump_json())

    def _call_gpt(self, message_history: list):

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        out = response.choices[0].message.content
        self.api_cost += response.usage.prompt_tokens * 0.21666 / 1000000 + response.usage.completion_tokens * 0.8667 / 1000000
        return out
    
    def _call_qwen7b(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )

        out = response.choices[0].message.content
        return out
    
    def _call_llama90(self, message_history: list):
        if self.task_type == "manip":
            response = self.model.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
                messages=message_history,
                response_format={"type": "json_object", "schema": ActionPlan_1_manip.model_json_schema()},
                temperature = temperature
            )
            out = response.choices[0].message.content
            
        else:
            response = self.model.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
                messages=message_history,
                response_format={"type": "json_object", "schema": ActionPlan_1.model_json_schema()},
                temperature = temperature
            )
            out = response.choices[0].message.content
        return out
    
    def _call_llama11b(self, message_history):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        out = response.choices[0].message.content
        return out
    

    def _call_qwen72b(self, message_history):
        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))
        
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )

        # easy to meet json errors
        out = response.choices[0].message.content
        out = out.replace("'",'"')
        out = out.replace('\"s ', "\'s ")
        out = out.replace('```json', '').replace('```', '')
        return out
    
    def _call_intern38b(self, message_history):

        # if not self.language_only:
        #     message_history = convert_format_2gemini(message_history)

        # no use, lmdeploy use support json schema only if it is pytorch-backended
        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            # response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )

        # easy to meet json errors
        out = response.choices[0].message.content
        out = out.replace("'",'"')
        out = out.replace('\"s ', "\'s ")
        out = out.replace('```json', '').replace('```', '')
        return out



if __name__ == "__main__":

    model = RemoteModel(
        'Qwen/Qwen2-VL-72B-Instruct', #'meta-llama/Llama-3.2-11B-Vision-Instruct',
        True #False
    )#'claude-3-5-sonnet-20241022, Qwen/Qwen2-VL-72B-Instruct, meta-llama/Llama-3.2-11B-Vision-Instruct


    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        

    base64_image = encode_image("../../evaluator/midlevel/output.png")
        
    messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                }
            },
            {
                "type": "text",
                "text":f"What do you think for this picture?? {template}?"
            },
            ],
        }
    ]

    response = model.respond(messages)
    print(response)

