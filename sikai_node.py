import torch
import os
import io
import openai
from PIL import Image
import requests
import numpy as np  # 用于处理 NumPy 数组
import random

from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class imageselect:
    CATEGORY = "Sikai_nodes/tools"
    @classmethod
    def INPUT_TYPES(s):
        return { "required":  { "images": ("IMAGE",), } }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "choose_image"
    def choose_image(self, images):
        brightness = list(torch.mean(image.flatten()).item() for image in images)
        brightest = brightness.index(max(brightness))
        result = images[brightest].unsqueeze(0)
        return (result,)


class SK_Random_File_Name:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": '', "multiline": False}),  # 输入文件夹路径
                "seed": ("INT", {"default": 0}),  # 全局 seed
                "fix_random": ("BOOLEAN", {"default": True})  # 控制是否固定随机性 (fix/random)
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("file_name_without_extension", "file_name_with_extension", "full_file_path")
    FUNCTION = "get_random_file_name"

    CATEGORY = "Sikai_nodes/tools"

    def get_random_file_name(self, folder_path, seed, fix_random):
        """
        随机读取文件夹中的一个文件，去掉后缀和完整文件名分别输出，此外输出完整路径。
        如果 fix_random 为 True，使用传入的 seed 值；否则使用真正的随机性。
        """
        # 如果 fix_random 为 True，使用 seed 进行固定的随机行为
        if fix_random:
            random.seed(seed)
        else:
            # 如果 fix_random 为 False，使用系统时间或其他不固定的随机行为
            random.seed(None)  # 让随机性基于系统时间

        # 检查文件夹是否存在
        if not os.path.isdir(folder_path):
            return ("Error: Folder not found", "", "")

        # 获取文件夹中的所有文件
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        # 如果文件夹为空，返回错误信息
        if not files:
            return ("Error: No files in folder", "", "")

        # 随机选择一个文件
        random_file = random.choice(files)

        # 获取文件名和扩展名
        file_name_without_extension = os.path.splitext(random_file)[0]
        file_name_with_extension = random_file

        # 获取完整路径
        full_file_path = os.path.join(folder_path, random_file)

        # 返回去除后缀的文件名、带后缀的完整文件名和完整文件路径
        return (file_name_without_extension, file_name_with_extension, full_file_path)
class SK_Text_String:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": True}),
            },
            "optional": {
                "text_b": ("STRING", {"default": '', "multiline": True}),
                "text_c": ("STRING", {"default": '', "multiline": True}),
                "text_d": ("STRING", {"default": '', "multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_text", "output_text_b", "output_text_c", "output_text_d")
    FUNCTION = "text_string"

    CATEGORY = "Sikai_nodes/tools"

    def text_string(self, text, text_b='', text_c='', text_d=''):
        """
        处理输入的四组文本，并分别返回。
        如果没有提供 text_b, text_c, text_d，使用默认空字符串。
        """
        # 返回四个字符串
        return (text, text_b, text_c, text_d)

class LoadAnyLLM:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM_NAME": ("STRING", {"default": "microsoft/Phi-3-mini-4k-instruct",
                "multiline": False
                }),

            }
        }

    RETURN_TYPES = ("MODEL", "TOKENIZER")
    RETURN_NAMES = ("LocalLLM", "tokenizer")
    FUNCTION = "load_model"
    CATEGORY = "Sikai_nodes/LLM"

    def load_model(self, LLM_NAME):
        model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
        return model, tokenizer

class GenerateTextFromLLM:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tokenizer": ("TOKENIZER",),
                "input_text": ("STRING", {"default": "how far is light year?", "multiline": True}),
                "max_length": ("INT", {"default": 50, "min": 10, "max": 512, "step": 10}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate_text"
    CATEGORY = "LocalLLM"

    def generate_text(self, model, tokenizer, input_text, max_length, temperature):
        # 使用分词器将输入文本编码为模型可以处理的格式
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

        # 使用模型生成文本
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,  # 允许生成的文本带有随机性
        )

        # 解码生成的文本为字符串
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return (generated_text,)


class OpenAIDAlleNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instruction": ("STRING", {  # 提供给 GPT 的专门化指导
                    "default": "you are a professional prompt engineer specializing in crafting detailed and precise prompts for stable diffusion models...",
                    "multiline": True
                }),
                "prompt": ("STRING", {  # 任务的具体内容
                    "default": "Describe this image",
                    "multiline": True
                }),
                "output_Image": ("BOOLEAN", {  # 用户选择是否输出图片
                    "default": False
                }),
                "input_Image": ("BOOLEAN", {  # 用户选择是否输入图片
                    "default": False
                }),
                "APIKey": ("STRING", {
                    "default": "hypr-lab-f5aBH69oHEJ03dD31G56T3BlbkFJfBjLrpUdAJiXLQYyMHUB"
                }),
                "EndpointADDRESS": ("STRING", {
                    "default": "https://api.hyprlab.io/v1/chat/completions"
                }),
                "ModelName": ("STRING", {
                    "default": "gpt-4"
                })
            },
            "optional": {
                "Referimage": ("IMAGE",)  # 可选图片输入
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("output_text", "output_image")
    FUNCTION = "generate_output"
    CATEGORY = "Sikai_nodes/LLM"

    def generate_output(self, instruction, prompt, output_Image, input_Image, Referimage, APIKey, ModelName, EndpointADDRESS):
        api_key = APIKey
        endpoint = EndpointADDRESS
        
        combined_prompt = f"{instruction}\n\n{prompt}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 准备请求数据
        data = {
            "model": ModelName,  # GPT 模型描述图像，但生成图像时需要调用 DALL·E 3
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt}
            ]
        }

        output_text = ""
        output_image = None

        # 将 Tensor 转换为 PIL 图像
        def convert_tensor_to_image(tensor_image):
            # 如果图像是 Tensor，将其转换为 PIL 图像
            if isinstance(tensor_image, torch.Tensor):
                # 首先移除多余的维度
                tensor_image = tensor_image.squeeze()  # 去掉 (1, 1, ...) 的维度

                # 确保数据的形状为 (height, width, channels)
                if tensor_image.dim() == 3 and tensor_image.size(2) == 3:
                    # 转换 Tensor 为 NumPy 数组，并确保类型为 uint8
                    np_image = tensor_image.mul(255).byte().cpu().numpy()
                    pil_image = Image.fromarray(np_image)
                    return pil_image
                else:
                    raise ValueError(f"Unexpected image shape: {tensor_image.size()}")
            return tensor_image

        # 情况 1: input_Image=True 且 output_Image=True -> 描述图片并生成图像
        if input_Image and output_Image:
            try:
                if Referimage is not None:  # 只处理非 None 的图片对象
                    # 如果 Referimage 是 Tensor，先转换为 PIL 图像
                    Referimage = convert_tensor_to_image(Referimage)
                    
                    # 转换图片为字节流
                    image_bytes = BytesIO()
                    Referimage.save(image_bytes, format='PNG')
                    image_data = image_bytes.getvalue()

                    # 调用 API 描述图片
                    description_response = requests.post(endpoint, headers=headers, json=data)
                    description_response.raise_for_status()
                    json_response = description_response.json()
                    output_text = json_response["choices"][0]["message"]["content"].strip()

                    # 生成图像（使用 DALL·E 3）
                    image_generation_response = requests.post(endpoint, headers=headers, json={
                        "model": "DALL·E 3",  # 使用 DALL·E 3 生成图像
                        "prompt": output_text,
                        "n": 1,
                        "size": "1024x1024"
                    })
                    image_generation_response.raise_for_status()
                    image_url = image_generation_response.json()['data'][0]['url']
                    img_data = requests.get(image_url).content
                    output_image = Image.open(BytesIO(img_data))
                    return (output_text, output_image)
                else:
                    return ("Error: No image provided for description.", None)
            except Exception as e:
                return (f"Error processing image and generating new image: {str(e)}", None)

        # 情况 2: input_Image=True 且 output_Image=False -> 描述图片，输出文本
        elif input_Image and not output_Image:
            try:
                if Referimage is not None:  # 确认图片非 None
                    Referimage = convert_tensor_to_image(Referimage)
                    
                    image_bytes = BytesIO()
                    Referimage.save(image_bytes, format='PNG')
                    image_data = image_bytes.getvalue()

                    # 调用 API 描述图片
                    description_response = requests.post(endpoint, headers=headers, json=data)
                    description_response.raise_for_status()
                    json_response = description_response.json()
                    output_text = json_response["choices"][0]["message"]["content"].strip()
                    return (output_text, None)
                else:
                    return ("Error: No image provided for description.", None)
            except Exception as e:
                return (f"Error processing image description: {str(e)}", None)

        # 情况 3: input_Image=False 且 output_Image=True -> 生成图像
        elif not input_Image and output_Image:
            try:
                # 根据 prompt 生成图像（使用 DALL·E 3）
                image_generation_response = requests.post(endpoint, headers=headers, json={
                    "model": "DALL·E 3",  # 使用 DALL·E 3 生成图像
                    "prompt": combined_prompt,
                    "n": 1,
                    "size": "1024x1024"
                })
                image_generation_response.raise_for_status()
                image_url = image_generation_response.json()['data'][0]['url']
                img_data = requests.get(image_url).content
                output_image = Image.open(BytesIO(img_data))
                return (None, output_image)
            except Exception as e:
                return (f"Error generating image: {str(e)}", None)

        # 情况 4: input_Image=False 且 output_Image=False -> 生成文本
        else:
            try:
                # 生成文本
                response = requests.post(endpoint, headers=headers, json=data)
                response.raise_for_status()
                json_response = response.json()
                output_text = json_response["choices"][0]["message"]["content"].strip()
                return (output_text, None)
            except Exception as e:
                return (f"Error generating text: {str(e)}", None)