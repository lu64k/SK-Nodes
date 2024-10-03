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
import torch.nn.functional as F
from sklearn.cluster import KMeans

from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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

class greyscaleblendNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_BW": ("IMAGE",),  # 输入图像A
                "image_Col": ("IMAGE",),  # 输入图像B
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lightness_overlay"

    CATEGORY = "sikai nodes/postprocess"

    def apply_lightness_overlay(self, image_BW: torch.Tensor, image_Col: torch.Tensor):
        """
        实现将图像A的明度叠加到图像B的色调上的效果。
        图像B保持色调不变，图像A提供明度变化。
        """
        # 将输入图像转换为 NumPy 格式
        image_a_np = (image_BW[0].cpu().numpy() * 255).astype(np.uint8)
        image_b_np = (image_Col[0].cpu().numpy() * 255).astype(np.uint8)

        # Step 1: 转换为 Lab 色彩空间
        lab_a = cv2.cvtColor(image_a_np, cv2.COLOR_RGB2LAB)  # 将图像A转换为Lab
        lab_b = cv2.cvtColor(image_b_np, cv2.COLOR_RGB2LAB)  # 将图像B转换为Lab

        # Step 2: 替换图像B的亮度通道为图像A的亮度通道
        # 完全用A图的L通道替换B图的L通道
        lab_b[:, :, 0] = lab_a[:, :, 0]

        # Step 3: 将修改后的Lab图像转换回RGB色彩空间
        combined_image = cv2.cvtColor(lab_b, cv2.COLOR_LAB2RGB)

        # 将结果转换为 torch.Tensor 并归一化
        combined_image_tensor = torch.tensor(combined_image).float() / 255
        return (combined_image_tensor.unsqueeze(0),)

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
