"""
赛博算命系统 - 命令行界面
"""

import argparse
import json
from fortune_teller import CyberFortuneTeller
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="🔮 赛博算命系统 - 基于CLIP的向量空间相似度计算",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 计算两个人的相似度
  python cli.py --name1 "张三" --face1 path/to/person1.jpg \\
                --name2 "李四" --face2 path/to/person2.jpg
  
  # 批量计算相似度矩阵
  python cli.py --batch --input batch_data.json --output similarity_matrix.json
  
  # 保存详细结果
  python cli.py --name1 "张三" --face1 person1.jpg \\
                --name2 "李四" --face2 person2.jpg \\
                --output result.json
        """
    )
    
    # 单人对算
    parser.add_argument("--name1", type=str, help="第一个人的名字")
    parser.add_argument("--face1", type=str, help="第一个人的照片路径")
    parser.add_argument("--name2", type=str, help="第二个人的名字")
    parser.add_argument("--face2", type=str, help="第二个人的照片路径")
    
    # 批量计算
    parser.add_argument("--batch", action="store_true", help="批量计算模式")
    parser.add_argument("--input", type=str, help="批量数据JSON文件路径")
    
    # 输出
    parser.add_argument("--output", type=str, help="结果输出路径（JSON格式）")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", 
                       help="CLIP模型名称")
    parser.add_argument("--device", type=str, default=None, 
                       help="设备 (cuda/cpu)，默认自动选择")
    
    args = parser.parse_args()
    
    # 初始化算命系统
    print("🔮 正在初始化赛博算命系统...")
    fortune_teller = CyberFortuneTeller(model_name=args.model, device=args.device)
    
    if args.batch:
        # 批量计算模式
        if not args.input:
            parser.error("批量模式需要 --input 参数")
        
        # 读取批量数据
        with open(args.input, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        person_list = [(item['name'], item['face_path']) for item in batch_data]
        
        # 计算相似度矩阵
        similarity_matrix = fortune_teller.batch_compute(person_list)
        
        # 保存结果
        result = {
            "similarity_matrix": similarity_matrix.tolist(),
            "person_list": [item['name'] for item in batch_data]
        }
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ 结果已保存到 {args.output}")
        else:
            print("\n相似度矩阵:")
            print(similarity_matrix)
    
    else:
        # 单人对算模式
        if not all([args.name1, args.face1, args.name2, args.face2]):
            parser.error("单人对算模式需要 --name1, --face1, --name2, --face2 参数")
        
        # 检查文件是否存在
        for face_path in [args.face1, args.face2]:
            if not Path(face_path).exists():
                parser.error(f"文件不存在: {face_path}")
        
        # 进行算命
        result = fortune_teller.tell_fortune(
            name1=args.name1,
            face1_path=args.face1,
            name2=args.name2,
            face2_path=args.face2
        )
        
        # 打印结果
        print("\n" + "="*60)
        print("🔮 赛博算命结果")
        print("="*60)
        print(f"👤 人物1: {result['person1']}")
        print(f"👤 人物2: {result['person2']}")
        print(f"\n📊 相似度分数:")
        print(f"  余弦相似度: {result['similarity_scores']['cosine']:.4f}")
        print(f"  点积相似度: {result['similarity_scores']['dot_product']:.4f}")
        print(f"  欧氏相似度: {result['similarity_scores']['euclidean_based']:.4f}")
        print(f"\n✨ 匹配度: {result['fortune']['compatibility']} ({result['fortune']['score']}/100)")
        print(f"\n📝 描述: {result['fortune']['description']}")
        print(f"\n💡 建议: {result['fortune']['advice']}")
        print(f"\n{result['fortune']['disclaimer']}")
        
        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 详细结果已保存到 {args.output}")


if __name__ == "__main__":
    main()

