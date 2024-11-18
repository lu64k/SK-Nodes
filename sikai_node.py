import torch
import os
import io
import time
import openai
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import requests
import numpy as np  # 用于处理 NumPy 数组
import random
import cv2
import kornia
import torch.nn.functional as F
from sklearn.cluster import KMeans

from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LoadAnyLLM:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM_NAME": ("STRING", {"default": "microsoft/Phi-3.5-mini-instruct",
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
                "Prompt": ("STRING", {"default": "how far is light year?", "multiline": True}),
                "system_instruction": ("STRING", {"default": "You are creating a prompt for Stable Diffusion to generate an image. First step: understand the input and generate a text prompt for the input. Second step: only respond in English with the prompt itself in phrase, but embellish it as needed but keep it under 200 tokens.", "multiline": True}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate_content"
    CATEGORY = "LocalLLM"

    def generate_content(self, model, tokenizer, Prompt, system_instruction, temperature):

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": Prompt},
        ]

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": temperature,
            "do_sample": False,
        }

        output = pipe(messages, **generation_args)
        textoutput = output[0]['generated_text']

        return (textoutput,)

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

class OpenAI_Text_Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instruction": ("STRING", {  # 提供给 GPT 的专门化指导
                    "default": "You are a professional prompt engineer specializing in crafting detailed and precise prompts for stable diffusion models...",
                    "multiline": True
                }),
                "prompt": ("STRING", {  # 任务的具体内容
                    "default": "Describe this image",
                    "multiline": True
                }),
                "backup_text": ("STRING", {  # 无法生成文本时输出的预置文本
                    "default": "A beautiful garden inside a crystal bottle",
                    "multiline": True
                }),
                "APIKey": ("STRING", {
                    "default": "hypr-lab-123456"
                }),
                "EndpointADDRESS": ("STRING", {
                    "default": "https://api.hyprlab.io/v1/chat/completions"
                }),
                "ModelName": ("STRING", {
                    "default": "gpt-4"
                })
            }
        }

    RETURN_TYPES = ("STRING",)  # 使用元组来定义返回类型
    RETURN_NAMES = ("output_text",)
    FUNCTION = "generate_output"
    CATEGORY = "Sikai_nodes/LLM"

    def generate_output(self, instruction, prompt, backup_text, APIKey, ModelName, EndpointADDRESS):
        api_key = APIKey
        endpoint = EndpointADDRESS

        # 组合系统提示与用户输入的 prompt
        combined_prompt = f"{instruction}\n\n{prompt}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 准备请求数据
        data = {
            "model": ModelName,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt}
            ]
        }

        # 尝试最多 3 次请求
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 发送请求
                response = requests.post(endpoint, headers=headers, json=data)
                response.raise_for_status()  # 检查请求是否成功
                json_response = response.json()

                # 获取生成的文本
                output_text = json_response["choices"][0]["message"]["content"].strip()
                return (output_text,)  # 如果成功，返回生成的文本

            except Exception as e:
                # 打印错误并等待一段时间再重试
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # 等待2秒后再重试
                else:
                    # 如果 3 次都失败，返回 backup_text
                    return (backup_text,)



class ToneLayerQuantize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "layers": ("INT", {  # 自定义分层数量
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "sigma_color": ("FLOAT", {
                    "default": 75.0,
                    "min": 0.0,
                    "max": 200.0,
                    "step": 0.1
                }),
                "sigma_space": ("FLOAT", {
                    "default": 75.0,
                    "min": 0.0,
                    "max": 200.0,
                    "step": 0.1
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "layered_quantize"

    CATEGORY = "sikai nodes/postprocess"

    def layered_quantize(self, image: torch.Tensor, layers: int = 5, sigma_color: float = 75.0, sigma_space: float = 75.0):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b]
            img = (tensor_image * 255).to(torch.uint8).numpy()

            # 应用双边滤波，平滑图像并保留边缘
            img_smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)

            # 图像分层：为每个通道自定义分层数量
            def layer_quantize_channel(channel, layers):
                # 找到每个通道的最大和最小值
                channel_min, channel_max = channel.min(), channel.max()
                # 定义层级间隔
                layer_step = (channel_max - channel_min) / layers
                # 将每个像素的值分配到某个层级
                channel = np.floor((channel - channel_min) / layer_step) * layer_step + channel_min
                return channel

            # 将每个颜色通道单独处理，应用分层
            for i in range(3):  # 对 R、G、B 三个通道分别操作
                img_smoothed[:, :, i] = layer_quantize_channel(img_smoothed[:, :, i], layers)

            # 将处理后的图像转换为 torch.Tensor 格式并归一化
            quantized_array = torch.tensor(img_smoothed).float() / 255
            result[b] = quantized_array

        return (result,)

class ColorTransferToneLayer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 分层后的图像
                "original_image": ("IMAGE",),  # 原始图像，用于颜色映射
                "layers": ("INT", {  # 设置分层数量
                    "default": 10,
                    "min": 2,
                    "max": 256,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_color_transfer"

    CATEGORY = "sikai nodes/postprocess"

    def apply_color_transfer(self, image: torch.Tensor, original_image: torch.Tensor, layers: int = 5):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        # 定义分层的步长，控制每个层级的颜色范围
        step = 255 // (layers - 1)

        for b in range(batch_size):
            # 原始图像和分层图像
            tensor_image = image[b]
            original_tensor_image = original_image[b]

            img = (tensor_image * 255).to(torch.uint8).numpy()
            original_img = (original_tensor_image * 255).to(torch.uint8).numpy()

            # 对 R, G, B 每个通道分别进行分层处理
            for i in range(3):  # 对应 RGB 三个通道
                # 将像素值映射到指定的层级
                img[:, :, i] = np.round(img[:, :, i] / step) * step

            # 使用颜色映射，将原始图像的颜色重新映射回分层图像
            img_colored = cv2.addWeighted(img, 0.5, original_img, 0.5, 0)

            # 将处理后的图像转换为 torch.Tensor 格式并归一化
            quantized_array = torch.tensor(img_colored).float() / 255
            result[b] = quantized_array

        return (result,)

class NaturalSaturationAdjust:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像，格式为 torch.Tensor
                "intensity": ("FLOAT", {  # 控制自然饱和度的调整强度
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_natural_saturation"

    CATEGORY = "sikai nodes/postprocess"
    def adjust_natural_saturation(self, image: torch.Tensor, intensity: float = 1.0):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            # 转换图像为 numpy 格式，图像值为 [0, 255] 的 8 位整数
            tensor_image = image[b]
            img = (tensor_image * 255).to(torch.uint8).numpy()

            # 转换到 HSV 颜色空间
            hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # 拆分 HSV 通道
            h, s, v = cv2.split(hsv_image)

            # 对饱和度通道 (S) 进行非线性调整
            s = s.astype(np.float32) / 255.0  # 正则化到 [0, 1] 范围

            # 自然饱和度公式：增强低饱和度，平滑高饱和度
            s = np.where(s < 0.5, s * (1 + intensity), s + (1 - s) * intensity * 0.3)

            # 限制饱和度值在 [0, 1] 之间
            s = np.clip(s, 0, 1)

            # 将饱和度恢复到 [0, 255] 范围并转换回 uint8 类型
            s = (s * 255).astype(np.uint8)

            # 合并调整后的 HSV 通道
            hsv_adjusted = cv2.merge([h, s, v])

            # 转换回 RGB 颜色空间
            rgb_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)

            # 将结果图像转换为 torch.Tensor 格式并归一化
            result[b] = torch.tensor(rgb_adjusted).float() / 255.0

        return (result,)

class ImageTracingNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "layers": ("INT", {  # 自定义分层数量
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "sigma_color": ("FLOAT", {
                    "default": 75.0,
                    "min": 0.0,
                    "max": 200.0,
                    "step": 0.1
                }),
                "sigma_space": ("FLOAT", {
                    "default": 75.0,
                    "min": 0.0,
                    "max": 200.0,
                    "step": 0.1
                }),
                "distance_threshold": ("FLOAT", {  # 欧氏距离阈值，用于判断颜色相似性
                    "default": 20.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 1.0
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "layered_quantize"

    CATEGORY = "sikai nodes/postprocess"

    def layered_quantize(self, image: torch.Tensor, layers: int = 5, sigma_color: float = 75.0, sigma_space: float = 75.0, distance_threshold: float = 20.0, use_gpu: bool = True):
        # 确保所有操作都在同一设备上
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image).to(device)

        for b in range(batch_size):
            tensor_image = image[b].to(device)
            img = (tensor_image * 255).to(torch.uint8).cpu().numpy()

            # Step 1: 使用 OpenCV 进行 RGB 到 Lab 颜色空间转换
            lab_image = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

            # Step 2: 应用双边滤波，平滑图像并保留边缘
            img_smoothed = cv2.bilateralFilter(lab_image, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)

            # Step 3: 计算颜色梯度 (基于 Lab 颜色空间的 L 通道)
            gradient_l = self.compute_gradient(torch.tensor(img_smoothed[:, :, 0:1]).to(device))  # 仅使用 L 通道计算梯度
            gradient_norm = F.normalize(gradient_l, p=2, dim=0)

            # 调整维度，确保 dynamic_threshold 是标量
            dynamic_threshold = (distance_threshold * (1 + gradient_norm)).squeeze().to(device)

            # Step 4: 基于颜色相似性并执行量化
            quantized_lab_image = self.color_quantize_lab_optimized(torch.tensor(img_smoothed).to(device), layers, dynamic_threshold, device)

            # Step 5: 将量化后的 Lab 图像转换回 RGB
            quantized_lab_image_np = quantized_lab_image.cpu().numpy()
            quantized_rgb_image = cv2.cvtColor(quantized_lab_image_np, cv2.COLOR_Lab2RGB)

            # 将结果转换为 torch.Tensor 格式并归一化
            quantized_tensor = torch.tensor(quantized_rgb_image).float().clamp(0, 255) / 255
            result[b] = quantized_tensor.to(device)

        return (result,)

    # 1. 计算颜色梯度（使用 PyTorch 进行加速）
    def compute_gradient(self, l_channel):
        """
        计算 L 通道的梯度，使用 Sobel 算子。
        """
        device = l_channel.device  # 获取 l_channel 所在设备

        if l_channel.dim() == 3:
            # 如果是 [height, width, channels]，首先添加一个 batch 维度
            l_channel = l_channel.unsqueeze(0).to(device)  # [1, height, width, channels]

        # 调整维度顺序为 [batch_size, channels, height, width]
        l_channel = l_channel.permute(0, 3, 1, 2).to(device)  # [1, 1, height, width]

        # **将 l_channel 转换为 float 类型**
        l_channel = l_channel.float()  # 转换为浮点数类型

        # 确保 Sobel 核也在同样的设备上
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

        # 卷积操作
        grad_x = F.conv2d(l_channel, sobel_x, padding=1)
        grad_y = F.conv2d(l_channel, sobel_y, padding=1)

        # 计算梯度
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        return gradient.squeeze(0)  # 去掉 batch 维度，返回 [channels, height, width]

    # 2. 批量处理的颜色量化（优化后的版本）
    def color_quantize_lab_optimized(self, img_lab, layers, dynamic_threshold, device):
        """
        使用批量处理优化颜色量化的计算，并减少逐像素的动态阈值比较。
        """
        # 将所有通道转换为 Float 类型
        l_channel = img_lab[:, :, 0].float().to(device)
        a_channel = img_lab[:, :, 1].float().to(device)
        b_channel = img_lab[:, :, 2].float().to(device)

        # 分层量化
        l_quant = self.quantize_channel(l_channel, l_channel.min(), l_channel.max(), layers)
        a_quant = self.quantize_channel(a_channel, a_channel.min(), a_channel.max(), layers)
        b_quant = self.quantize_channel(b_channel, b_channel.min(), b_channel.max(), layers)

        # 使用批量操作计算颜色相似性
        dist_l = (l_quant - l_channel) ** 2
        dist_a = (a_quant - a_channel) ** 2
        dist_b = (b_quant - b_channel) ** 2
        dist = torch.sqrt(dist_l + dist_a + dist_b)

        # 使用动态阈值进行批量比较
        mask = dist < dynamic_threshold

        # 将量化后的结果应用到符合条件的像素上，确保数据类型一致
        l_channel[mask] = l_quant[mask]
        a_channel[mask] = a_quant[mask]
        b_channel[mask] = b_quant[mask]

        # 组合量化后的通道
        img_lab[:, :, 0] = l_channel
        img_lab[:, :, 1] = a_channel
        img_lab[:, :, 2] = b_channel

        return img_lab

    # 3. 通道量化
    def quantize_channel(self, channel, min_val, max_val, layers):
        """
        根据指定层数对通道进行量化。
        """
        layer_step = (max_val - min_val) / layers
        return torch.floor((channel - min_val) / layer_step) * layer_step + min_val

import torch
import kornia

class greyscaleblendNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_BW": ("IMAGE",),  # Input grayscale image (batch)
                "image_Col": ("IMAGE",),  # Input color image (batch)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lightness_overlay"
    CATEGORY = "sikai nodes/postprocess"

    def apply_lightness_overlay(self, image_BW: torch.Tensor, image_Col: torch.Tensor):
        """
        Applies the lightness from image_BW to the color of image_Col.
        image_Col retains its color, and image_BW provides the lightness variations.
        """
        # Ensure images are on the same device
        device = image_BW.device
        image_Col = image_Col.to(device)

        # Print input image shapes
        print(f"Original image_BW shape: {image_BW.shape}")
        print(f"Original image_Col shape: {image_Col.shape}")

        # If images are in NHWC format, convert to NCHW
        if image_BW.shape[-1] in [1, 3, 4]:
            image_BW = image_BW.permute(0, 3, 1, 2)
            image_Col = image_Col.permute(0, 3, 1, 2)
            print("Permuted images to [batch_size, channels, height, width] format.")

        # Print adjusted shapes
        print(f"image_BW shape after permute: {image_BW.shape}")
        print(f"image_Col shape after permute: {image_Col.shape}")

        # Ensure images have 3 channels
        if image_BW.shape[1] == 1:
            image_BW = image_BW.repeat(1, 3, 1, 1)
            print(f"Expanded image_BW to 3 channels: {image_BW.shape}")
        elif image_BW.shape[1] == 4:
            image_BW = image_BW[:, :3, :, :]
            print(f"Trimmed image_BW to 3 channels: {image_BW.shape}")

        if image_Col.shape[1] == 1:
            image_Col = image_Col.repeat(1, 3, 1, 1)
            print(f"Expanded image_Col to 3 channels: {image_Col.shape}")
        elif image_Col.shape[1] == 4:
            image_Col = image_Col[:, :3, :, :]
            print(f"Trimmed image_Col to 3 channels: {image_Col.shape}")

        # Ensure images have the same dtype and range
        image_BW = image_BW.float() / 255.0 if image_BW.max() > 1 else image_BW.float()
        image_Col = image_Col.float() / 255.0 if image_Col.max() > 1 else image_Col.float()

        # Convert images to LAB color space
        lab_bw = kornia.color.rgb_to_lab(image_BW)
        lab_col = kornia.color.rgb_to_lab(image_Col)

        # Replace the L channel of lab_col with that of lab_bw
        lab_col[:, 0:1, :, :] = lab_bw[:, 0:1, :, :]

        # Convert back to RGB
        combined_img = kornia.color.lab_to_rgb(lab_col)

        # Clamp values to [0, 1]
        combined_img = torch.clamp(combined_img, 0.0, 1.0)

        # Convert back to NHWC format if needed
        combined_img = combined_img.permute(0, 2, 3, 1)
        print(f"Output image shape: {combined_img.shape}")
        print(f"Output image dtype: {combined_img.dtype}")

        # **Move tensor to CPU before any NumPy conversion**
        combined_img_cpu = combined_img.cpu()

        # If you need to convert to NumPy array
        output_img = combined_img_cpu.numpy()
        print(f"Output image dtype after numpy conversion: {output_img.dtype}")

        # Return the combined image tensor (still on CPU)
        return (combined_img_cpu,)


class LoadNemotron:
    def __init__(self):
        self.conversation_history = ""  # 初始化对话历史
        self.rounds = 0


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_instruction": ("STRING", {"default": "You are playing the role of a naughty little rat", "multiline": True}),
                "prompt": ("STRING", {"default": "How are you?", "multiline": True}),
                "extra_id_0": ("STRING", {"default": "<extra_id_0>"}),  # System 标记
                "extra_id_1": ("STRING", {"default": "<extra_id_1>"}),  # User 和 Assistant 标记
                "Role_sysName": ("STRING", {"default": "actor"}),
                "Role_userName": ("STRING", {"default": "User"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("conversation_output",)
    FUNCTION = "generate_conversation"
    CATEGORY = "LocalLLM"

    def generate_conversation(self, system_instruction, prompt, extra_id_0, extra_id_1,Role_sysName, Role_userName):
        # Load tokenizer and model from NVIDIA's Nemotron
        tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-Mini-4B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("nvidia/Nemotron-Mini-4B-Instruct")

        # 新的用户输入
        new_turn = f"""
{extra_id_1}{Role_userName}
{prompt}
{extra_id_1}{Role_sysName}\n
"""

        # 如果是第一轮对话，加入 system_instruction
        if self.rounds == 0:
            temp_conversation = f"""
{extra_id_0}System
{system_instruction}
""" + self.conversation_history + new_turn
        else:
            temp_conversation = self.conversation_history + new_turn

        # Tokenize the conversation (including system instruction and conversation history)
        inputs = tokenizer(temp_conversation, return_tensors="pt")

        # Generate response from the model
        output_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode the generated output
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 更新 conversation_history，累加上一轮回复文本和新输入的用户 prompt
        self.conversation_history = f"{generated_text}\n"

        # 增加对话轮次
        self.rounds += 1

        # 返回当前的对话历史
        return (self.conversation_history,)

