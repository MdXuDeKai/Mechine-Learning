# Streamlit Community Cloud 部署指南

## 📦 部署前准备

### 1. 确保所有文件就绪

检查以下文件是否存在于 `web/` 目录：
- ✅ `app.py` - 主应用文件
- ✅ `requirements.txt` - 依赖列表
- ✅ `README.md` - 项目说明
- ✅ `model_AdaBoost.pkl` - 训练好的模型（必需）
- ✅ `final_features.pkl` - 特征列表（可选）

### 2. 检查模型文件大小

```bash
# 检查文件大小
ls -lh web/model_AdaBoost.pkl
```

**注意**：如果模型文件 > 100MB，建议使用 Git LFS 或外部存储。

### 3. 测试本地运行

在部署前，确保应用可以在本地正常运行：

```bash
cd web
streamlit run app.py
```

访问 `http://localhost:8501` 测试所有功能。

## 🚀 部署步骤

### 方法1：通过Streamlit Community Cloud网站

1. **访问部署网站**
   - 打开 https://share.streamlit.io/
   - 使用GitHub账户登录

2. **创建新应用**
   - 点击 "New app" 按钮
   - 选择你的GitHub账户和仓库

3. **配置应用**
   - **Repository**: 选择包含 `web/` 文件夹的仓库
   - **Branch**: 选择主分支（通常是 `main` 或 `master`）
   - **Main file path**: 输入 `web/app.py`
   - **App URL**: 可以自定义（可选）

4. **部署**
   - 点击 "Deploy" 按钮
   - 等待部署完成（通常需要2-5分钟）

5. **查看日志**
   - 如果部署失败，点击 "Manage app" → "Logs" 查看错误信息

### 方法2：通过GitHub Actions（高级）

如果你熟悉GitHub Actions，可以设置自动部署。

## 🔧 常见问题解决

### 问题1：模型文件找不到

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'model_AdaBoost.pkl'
```

**解决方案**：
1. 确保模型文件已提交到GitHub仓库
2. 检查文件路径是否正确
3. 如果文件太大，使用Git LFS：
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes web/model_AdaBoost.pkl
git commit -m "Add model with LFS"
git push
```

### 问题2：依赖安装失败

**错误信息**：
```
ERROR: Could not find a version that satisfies the requirement...
```

**解决方案**：
1. 检查 `requirements.txt` 中的包版本
2. 确保所有依赖都是公开可用的
3. 尝试固定版本号，例如：
```
streamlit==1.28.0
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.1
```

### 问题3：内存不足

**错误信息**：
```
MemoryError: Unable to allocate array
```

**解决方案**：
1. 优化模型大小（如果可能）
2. 使用模型量化技术
3. 考虑使用外部存储加载模型

### 问题4：应用加载缓慢

**解决方案**：
1. 使用 `@st.cache_resource` 缓存模型加载
2. 优化代码，减少不必要的计算
3. 考虑使用更轻量的模型格式

## 📝 部署后检查清单

- [ ] 应用可以正常访问
- [ ] 模型可以成功加载
- [ ] 输入表单正常工作
- [ ] 预测功能正常
- [ ] 所有特征输入都能正确处理
- [ ] 衍生指标计算正确
- [ ] 响应式设计在不同设备上正常显示

## 🔄 更新应用

当你更新代码后：

1. **提交更改到GitHub**
```bash
git add .
git commit -m "Update app"
git push
```

2. **在Streamlit Cloud中重新部署**
   - 访问你的应用管理页面
   - 点击 "Reboot app" 或等待自动重新部署

## 📊 监控和维护

### 查看应用使用情况

在Streamlit Cloud管理页面可以查看：
- 访问量统计
- 错误日志
- 性能指标

### 定期检查

- 每周检查一次应用是否正常运行
- 监控错误日志
- 根据用户反馈更新功能

## 🆘 获取帮助

如果遇到问题：
1. 查看 Streamlit Cloud 文档：https://docs.streamlit.io/streamlit-community-cloud
2. 查看应用日志找出具体错误
3. 在 GitHub Issues 中提问

## 📌 重要提示

⚠️ **模型文件大小限制**：
- Streamlit Cloud 对单个文件有大小限制
- 如果模型文件 > 100MB，建议使用 Git LFS 或外部存储

⚠️ **公开仓库要求**：
- Streamlit Community Cloud 免费版只支持公开仓库
- 如果需要私有部署，考虑使用 Streamlit Cloud for Teams

⚠️ **资源限制**：
- 免费版有CPU和内存限制
- 如果应用使用资源过多，可能需要升级
