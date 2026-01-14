# 阑尾炎复杂性预测Web应用

基于AdaBoost机器学习模型的Streamlit Web应用，用于预测阑尾炎复杂性（复杂阑尾炎 vs 单纯阑尾炎）。

## 📋 项目信息

- **模型算法**: AdaBoost
- **测试集AUC**: 0.828
- **特征数量**: 7个（经过特征筛选后的最优特征集）
- **最佳阈值**: 0.4963
- **部署平台**: Streamlit Community Cloud

### 7个特征列表
1. `preop_crp` - 术前CRP（mg/L）
2. `MLR` - 单核细胞/淋巴细胞比值（自动计算）
3. `NLR` - 中性粒细胞/淋巴细胞比值（自动计算）
4. `diameter` - 阑尾直径（mm）
5. `weight` - 体重（kg）
6. `preop_plt` - 术前血小板（×10⁹/L）
7. `NMLR` - 中性粒细胞/(单核细胞+淋巴细胞)比值（自动计算）

## 🚀 快速开始

### 本地运行

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **准备模型文件**
   - 将训练好的模型文件 `final_model.pkl` 复制到 `web/` 目录
   - 或者运行 `python setup.py` 自动复制模型文件

3. **运行应用**
```bash
streamlit run app.py
```

应用将在浏览器中自动打开（通常是 `http://localhost:8501`）

## ☁️ 部署到Streamlit Community Cloud

### 前置要求

1. **GitHub账户**：确保你的代码已推送到GitHub公开仓库
2. **模型文件**：将模型文件包含在仓库中或使用其他存储方式

### 部署步骤

1. **准备仓库结构**
```
your-repo/
├── web/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   └── model_AdaBoost.pkl  # 模型文件
└── README.md
```

2. **访问Streamlit Community Cloud**
   - 访问 https://share.streamlit.io/
   - 使用GitHub账户登录

3. **部署应用**
   - 点击 "New app"
   - 选择你的GitHub仓库
   - 设置以下参数：
     - **Repository**: 你的GitHub仓库
     - **Branch**: main (或你的主分支)
     - **Main file path**: `web/app.py`
   - 点击 "Deploy"

4. **等待部署完成**
   - 首次部署可能需要几分钟
   - 部署完成后会显示应用URL

### 模型文件处理

如果模型文件较大（>100MB），可以考虑以下方案：

1. **使用Git LFS**
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add web/model_AdaBoost.pkl
git commit -m "Add model file with LFS"
```

2. **使用外部存储**
   - 将模型文件上传到云存储（如Google Drive, AWS S3等）
   - 在应用启动时下载模型文件
   - 修改 `app.py` 中的 `load_model()` 函数

3. **使用Hugging Face Model Hub**
   - 将模型上传到Hugging Face
   - 在应用中从Hugging Face加载

## 📁 文件说明

- `app.py`: Streamlit主应用文件
- `requirements.txt`: Python依赖包列表
- `README.md`: 项目说明文档
- `final_model.pkl`: 训练好的最终模型（需要从 `结果/` 目录复制）
- `setup.py`: 自动设置脚本（复制模型文件）

## 🔧 配置说明

### 模型路径

应用会自动尝试从以下路径加载模型：
1. `../结果/final_model.pkl` (本地开发，优先)
2. `结果/final_model.pkl` (相对路径)
3. `./final_model.pkl` (当前目录)
4. `../结果/model_AdaBoost.pkl` (备用路径)
5. `结果/model_AdaBoost.pkl` (备用路径)
6. `./model_AdaBoost.pkl` (备用路径)

如果模型文件在其他位置，请修改 `app.py` 中的 `load_model()` 函数。

### 特征列表

应用使用固定的7个特征（已在代码中定义），无需额外的特征列表文件。

## 📊 功能特性

- ✅ 交互式输入表单（仅需输入基础检验指标）
- ✅ 自动计算衍生指标（MLR, NLR, NMLR）
- ✅ 实时风险预测（使用最佳阈值0.4963）
- ✅ 可视化预测结果
- ✅ 响应式设计
- ✅ 无需数据标准化（AdaBoost基于树，不需要标准化）

## 🐛 故障排除

### 模型加载失败

如果遇到模型加载错误：
1. 检查模型文件路径是否正确
2. 确保模型文件与训练时使用的scikit-learn版本兼容
3. 检查文件权限

### 依赖安装问题

如果遇到依赖安装问题：
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### 部署问题

如果Streamlit Cloud部署失败：
1. 检查 `requirements.txt` 中的包版本是否兼容
2. 确保所有必需文件都在仓库中
3. 查看Streamlit Cloud的日志输出

## 📝 更新日志

### v1.0.0 (2024)
- 初始版本
- 基于AdaBoost模型
- 使用7个最优特征（经过特征筛选）
- 自动计算衍生指标（MLR, NLR, NMLR）
- 使用最佳阈值0.4963进行分类
- 无需数据标准化

## 📄 许可证

本项目仅供研究使用。

## ⚠️ 免责声明

本预测系统仅供参考，不能替代专业医疗诊断。所有医疗决策应由专业医生做出。

## 📧 联系方式

如有问题或建议，请通过GitHub Issues联系。
