# 🚀 快速开始指南

## 第一步：准备模型文件

运行设置脚本复制模型文件：

```bash
cd web
python setup.py
```

或者手动复制：
```bash
# Windows
copy ..\结果\final_model.pkl .

# Linux/Mac
cp ../结果/final_model.pkl .
```

**注意**：应用使用固定的7个特征，无需复制特征列表文件。

## 第二步：安装依赖

```bash
pip install -r requirements.txt
```

## 第三步：运行应用

```bash
streamlit run app.py
```

应用会自动在浏览器中打开（通常是 `http://localhost:8501`）

## 第四步：测试应用

1. 在输入表单中填写患者信息
2. 点击"开始预测"按钮
3. 查看预测结果

## 📦 部署到Streamlit Cloud

### 前提条件

1. ✅ 代码已推送到GitHub公开仓库
2. ✅ 模型文件已包含在仓库中（或使用Git LFS）
3. ✅ 所有依赖都在 `requirements.txt` 中

### 部署步骤

1. 访问 https://share.streamlit.io/
2. 使用GitHub账户登录
3. 点击 "New app"
4. 配置：
   - **Repository**: 你的GitHub仓库
   - **Branch**: main
   - **Main file path**: `web/app.py`
5. 点击 "Deploy"
6. 等待部署完成（2-5分钟）

### 部署后

- 应用URL格式：`https://your-app-name.streamlit.app`
- 可以分享给其他人使用
- 每次代码更新后会自动重新部署

## ⚠️ 常见问题

### Q: 模型文件找不到？

A: 确保运行了 `setup.py` 或手动复制了 `final_model.pkl` 到 `web/` 目录。应用会优先查找 `final_model.pkl`，如果不存在会尝试查找 `model_AdaBoost.pkl`。

### Q: 依赖安装失败？

A: 尝试：
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Q: 应用运行但预测失败？

A: 检查：
1. 模型文件是否正确加载
2. 输入数据格式是否正确
3. 查看终端错误信息

### Q: 部署到Streamlit Cloud失败？

A: 检查：
1. 仓库是否为公开仓库
2. `requirements.txt` 中的包版本是否兼容
3. 模型文件是否太大（>100MB需要Git LFS）
4. 查看Streamlit Cloud的日志输出

## 📞 需要帮助？

查看详细文档：
- `README.md` - 完整项目说明
- `DEPLOY.md` - 详细部署指南
