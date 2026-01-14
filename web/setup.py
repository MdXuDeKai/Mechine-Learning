"""
设置脚本：复制模型文件到web目录
"""
import os
import shutil
from pathlib import Path

def setup_models():
    """复制模型文件到web目录"""
    base_dir = Path(__file__).parent.parent
    web_dir = Path(__file__).parent
    
    # 模型文件路径（优先使用final_model.pkl）
    model_source_final = base_dir / "结果" / "final_model.pkl"
    model_source_ada = base_dir / "结果" / "model_AdaBoost.pkl"
    model_dest = web_dir / "final_model.pkl"
    
    # 特征文件路径
    feature_source = base_dir / "结果" / "final_features.pkl"
    feature_dest = web_dir / "final_features.pkl"
    
    print("=" * 60)
    print("设置模型文件")
    print("=" * 60)
    
    # 复制模型文件（优先使用final_model.pkl）
    if model_source_final.exists():
        shutil.copy2(model_source_final, model_dest)
        print(f"✅ 已复制模型文件: {model_dest}")
    elif model_source_ada.exists():
        shutil.copy2(model_source_ada, model_dest)
        print(f"✅ 已复制模型文件（从model_AdaBoost.pkl）: {model_dest}")
    else:
        print(f"⚠️  模型文件不存在: {model_source_final} 或 {model_source_ada}")
        print("   请确保模型文件在 '结果/final_model.pkl' 或 '结果/model_AdaBoost.pkl'")
    
    # 复制特征文件
    if feature_source.exists():
        shutil.copy2(feature_source, feature_dest)
        print(f"✅ 已复制特征文件: {feature_dest}")
    else:
        print(f"ℹ️  特征文件不存在: {feature_source}")
        print("   应用将使用默认特征列表")
    
    print("=" * 60)
    print("设置完成！")
    print("=" * 60)
    print("\n下一步：")
    print("1. 运行 'streamlit run app.py' 测试应用")
    print("2. 或按照 README.md 中的说明部署到Streamlit Cloud")

if __name__ == "__main__":
    setup_models()
