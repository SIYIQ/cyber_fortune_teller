"""
赛博算命系统 - 使用示例
"""

from fortune_teller import CyberFortuneTeller
import json
from pathlib import Path


def example_single():
    """示例1: 单人对算"""
    print("="*60)
    print("示例1: 单人对算")
    print("="*60)
    
    # 初始化系统
    fortune_teller = CyberFortuneTeller()
    
    # 注意：需要替换为实际的图像路径
    # result = fortune_teller.tell_fortune(
    #     name1="张三",
    #     face1_path="path/to/person1.jpg",
    #     name2="李四",
    #     face2_path="path/to/person2.jpg"
    # )
    # 
    # print(f"\n相似度: {result['similarity_scores']['cosine']:.4f}")
    # print(f"匹配度: {result['fortune']['compatibility']}")
    
    print("💡 请替换为实际的图像路径后运行")


def example_batch():
    """示例2: 批量计算相似度矩阵"""
    print("\n" + "="*60)
    print("示例2: 批量计算")
    print("="*60)
    
    fortune_teller = CyberFortuneTeller()
    
    # 准备数据
    person_list = [
        ("张三", "path/to/person1.jpg"),
        ("李四", "path/to/person2.jpg"),
        ("王五", "path/to/person3.jpg"),
    ]
    
    # 计算相似度矩阵
    # similarity_matrix = fortune_teller.batch_compute(person_list)
    # print(f"\n相似度矩阵:\n{similarity_matrix}")
    
    print("💡 请替换为实际的图像路径后运行")


def example_custom_text_prompts():
    """示例3: 自定义文本提示词"""
    print("\n" + "="*60)
    print("示例3: 自定义编码方式")
    print("="*60)
    
    fortune_teller = CyberFortuneTeller()
    
    # 可以修改 fortune_teller.py 中的 encode_name 方法
    # 尝试不同的文本提示词，比如：
    # - f"a photo of {name}"
    # - f"{name}'s portrait"
    # - f"the person {name}"
    
    print("💡 可以修改 encode_name 方法中的文本提示词来实验不同效果")


def example_save_results():
    """示例4: 保存结果到JSON"""
    print("\n" + "="*60)
    print("示例4: 保存结果")
    print("="*60)
    
    fortune_teller = CyberFortuneTeller()
    
    # result = fortune_teller.tell_fortune(
    #     name1="张三",
    #     face1_path="path/to/person1.jpg",
    #     name2="李四",
    #     face2_path="path/to/person2.jpg"
    # )
    # 
    # # 保存结果
    # with open("result.json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=2)
    # 
    # print("✅ 结果已保存到 result.json")
    
    print("💡 请替换为实际的图像路径后运行")


if __name__ == "__main__":
    print("🔮 赛博算命系统 - 使用示例\n")
    
    example_single()
    example_batch()
    example_custom_text_prompts()
    example_save_results()
    
    print("\n" + "="*60)
    print("更多用法请参考 README.md")
    print("="*60)

