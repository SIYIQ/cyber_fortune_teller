"""
赛博算命系统 - 基于CLIP的向量空间相似度计算
核心思想：将两个人的名字和脸的向量拼接后，计算在高维空间中的相似度
"""

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
from typing import Tuple, Dict, Optional
import os


class CyberFortuneTeller:
    """赛博算命系统主类"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        """
        初始化赛博算命系统
        
        Args:
            model_name: CLIP模型名称
            device: 设备（cuda/cpu），如果为None则自动选择
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔮 正在加载CLIP模型到 {self.device}...")
        
        # 加载CLIP模型和处理器
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # 冻结CLIP参数（我们只是用它来提取特征）
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        self.embed_dim = self.model.config.projection_dim  # 通常是512
        print(f"✅ CLIP模型加载完成！嵌入维度: {self.embed_dim}")
    
    def encode_name(self, name: str) -> torch.Tensor:
        """
        编码名字（文本）为向量
        
        Args:
            name: 人名
            
        Returns:
            名字的向量表示 [embed_dim]
        """
        # 构造文本提示，可以尝试不同的提示词
        text_prompts = [
            f"a person named {name}",
            f"{name}",
            f"the face of {name}",
        ]
        
        # 对所有提示词编码并取平均（更稳定）
        text_embeds = []
        for prompt in text_prompts:
            inputs = self.processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_outputs = self.model.get_text_features(**inputs)
                text_embeds.append(text_outputs)
        
        # 平均所有提示词的嵌入
        text_embed = torch.mean(torch.cat(text_embeds, dim=0), dim=0)
        return text_embed
    
    def encode_face(self, image_path: str) -> torch.Tensor:
        """
        编码人脸图像为向量
        
        Args:
            image_path: 图像路径
            
        Returns:
            人脸的向量表示 [embed_dim]
        """
        # 加载图像
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path  # 假设已经是PIL Image
        
        # 处理图像
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 编码
        with torch.no_grad():
            image_embed = self.model.get_image_features(**inputs)
            image_embed = image_embed.squeeze(0)  # 移除batch维度
        
        return image_embed
    
    def encode_person(self, name: str, face_image_path: str) -> torch.Tensor:
        """
        编码一个人：将名字和脸的向量拼接
        
        Args:
            name: 人名
            face_image_path: 人脸图像路径
            
        Returns:
            拼接后的向量 [embed_dim * 2]
        """
        name_vec = self.encode_name(name)
        face_vec = self.encode_face(face_image_path)
        
        # 拼接向量
        person_vec = torch.cat([name_vec, face_vec], dim=0)
        
        return person_vec
    
    def compute_similarity(self, person1_vec: torch.Tensor, person2_vec: torch.Tensor, 
                          method: str = "cosine") -> float:
        """
        计算两个人在高维空间中的相似度
        
        Args:
            person1_vec: 第一个人的向量
            person2_vec: 第二个人的向量
            method: 相似度计算方法 ("cosine", "euclidean", "dot")
            
        Returns:
            相似度分数
        """
        # 归一化向量（对余弦相似度很重要）
        person1_vec = F.normalize(person1_vec, p=2, dim=0)
        person2_vec = F.normalize(person2_vec, p=2, dim=0)
        
        if method == "cosine":
            similarity = F.cosine_similarity(person1_vec.unsqueeze(0), 
                                           person2_vec.unsqueeze(0), dim=1).item()
        elif method == "dot":
            similarity = torch.dot(person1_vec, person2_vec).item()
        elif method == "euclidean":
            # 欧氏距离转换为相似度（距离越小，相似度越高）
            distance = torch.norm(person1_vec - person2_vec).item()
            similarity = 1.0 / (1.0 + distance)  # 转换为0-1之间的相似度
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        return similarity
    
    def tell_fortune(self, name1: str, face1_path: str, 
                    name2: str, face2_path: str) -> Dict:
        """
        赛博算命：计算两个人的相似度并生成"算命"结果
        
        Args:
            name1: 第一个人的名字
            face1_path: 第一个人的照片路径
            name2: 第二个人的名字
            face2_path: 第二个人的照片路径
            
        Returns:
            包含相似度和算命结果的字典
        """
        print(f"\n🔮 正在为 {name1} 和 {name2} 进行赛博算命...")
        
        # 编码两个人
        person1_vec = self.encode_person(name1, face1_path)
        person2_vec = self.encode_person(name2, face2_path)
        
        # 计算多种相似度
        cosine_sim = self.compute_similarity(person1_vec, person2_vec, method="cosine")
        dot_sim = self.compute_similarity(person1_vec, person2_vec, method="dot")
        euclidean_sim = self.compute_similarity(person1_vec, person2_vec, method="euclidean")
        
        # 生成算命结果
        fortune_result = self._generate_fortune_result(cosine_sim, name1, name2)
        
        result = {
            "person1": name1,
            "person2": name2,
            "similarity_scores": {
                "cosine": cosine_sim,
                "dot_product": dot_sim,
                "euclidean_based": euclidean_sim,
            },
            "fortune": fortune_result,
        }
        
        return result
    
    def _generate_fortune_result(self, similarity: float, name1: str, name2: str) -> Dict:
        """
        根据相似度生成"算命"结果
        
        Args:
            similarity: 相似度分数（0-1）
            name1: 第一个人的名字
            name2: 第二个人的名字
            
        Returns:
            算命结果字典
        """
        # 将相似度映射到0-100的分数
        score = int(similarity * 100)
        
        # 根据相似度区间生成不同的"算命"结果
        if similarity >= 0.9:
            compatibility = "天作之合"
            description = f"{name1}和{name2}在CLIP的高维空间中高度相似，可能是灵魂伴侣！"
            advice = "建议：你们在多个维度上都非常匹配，可以深入发展关系。"
        elif similarity >= 0.8:
            compatibility = "非常匹配"
            description = f"{name1}和{name2}在向量空间中表现出很强的相似性。"
            advice = "建议：你们有很多共同点，可以尝试更多互动。"
        elif similarity >= 0.7:
            compatibility = "较为匹配"
            description = f"{name1}和{name2}在语义空间中有一定的相似度。"
            advice = "建议：虽然有一些差异，但仍有发展的潜力。"
        elif similarity >= 0.5:
            compatibility = "中等匹配"
            description = f"{name1}和{name2}在向量空间中的相似度处于中等水平。"
            advice = "建议：需要更多了解才能判断是否合适。"
        elif similarity >= 0.3:
            compatibility = "不太匹配"
            description = f"{name1}和{name2}在语义空间中差异较大。"
            advice = "建议：可能需要更多努力才能建立联系。"
        else:
            compatibility = "差异较大"
            description = f"{name1}和{name2}在CLIP的高维空间中表现出明显差异。"
            advice = "建议：差异可能带来互补，也可能带来挑战。"
        
        return {
            "compatibility": compatibility,
            "score": score,
            "description": description,
            "advice": advice,
            "disclaimer": "⚠️ 本结果仅供娱乐，基于CLIP模型的向量空间相似度计算，不代表真实的人际关系。"
        }
    
    def batch_compute(self, person_list: list) -> np.ndarray:
        """
        批量计算多个人之间的相似度矩阵
        
        Args:
            person_list: 列表，每个元素是 (name, face_path) 元组
            
        Returns:
            相似度矩阵 [n, n]
        """
        n = len(person_list)
        similarity_matrix = np.zeros((n, n))
        
        print(f"📊 正在计算 {n} 个人之间的相似度矩阵...")
        
        # 编码所有人
        person_vectors = []
        for name, face_path in person_list:
            vec = self.encode_person(name, face_path)
            person_vectors.append(vec)
        
        # 计算两两相似度
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.compute_similarity(person_vectors[i], person_vectors[j])
                    similarity_matrix[i, j] = sim
        
        return similarity_matrix


def main():
    """示例用法"""
    # 初始化算命系统
    fortune_teller = CyberFortuneTeller()
    
    # 示例：计算两个人的相似度
    # 注意：需要提供真实的图像路径
    print("\n" + "="*60)
    print("赛博算命系统示例")
    print("="*60)
    
    # 这里需要替换为实际的图像路径
    # result = fortune_teller.tell_fortune(
    #     name1="张三",
    #     face1_path="path/to/person1.jpg",
    #     name2="李四",
    #     face2_path="path/to/person2.jpg"
    # )
    # 
    # print("\n" + "="*60)
    # print("算命结果")
    # print("="*60)
    # print(f"相似度分数: {result['similarity_scores']['cosine']:.4f}")
    # print(f"匹配度: {result['fortune']['compatibility']}")
    # print(f"描述: {result['fortune']['description']}")
    # print(f"建议: {result['fortune']['advice']}")
    # print(f"\n{result['fortune']['disclaimer']}")
    
    print("\n💡 使用方法：")
    print("1. 准备两个人的照片")
    print("2. 调用 fortune_teller.tell_fortune(name1, face1_path, name2, face2_path)")
    print("3. 查看相似度和算命结果")


if __name__ == "__main__":
    main()

