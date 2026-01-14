# Mechine-Learning
# 🏥 Preoperative Prediction Model for Complicated Appendicitis

A machine learning-based preoperative prediction system for complicated appendicitis (gangrenous/perforated/periappendicitis), using AdaBoost algorithm with rigorous feature selection and external validation.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

## 📋 Project Overview

This project develops a machine learning-based preoperative prediction model for complicated appendicitis, aiming to help clinicians identify high-risk patients before surgery and optimize treatment decisions.

### Key Features

- ✅ **High Accuracy**: Test set AUC = 0.828, Sensitivity 92.7%, Specificity 65.2%
- ✅ **Feature Optimization**: Selected 7 optimal features from 23 candidate features
- ✅ **Rigorous Validation**: Includes external validation and temporal validation to ensure model generalizability
- ✅ **Clinical Utility**: All features are preoperatively available, no postoperative information required
- ✅ **Web Application**: Provides Streamlit Web interface for easy clinical use

## 🎯 Model Performance

### Final Model (AdaBoost, 7 Features)

| Metric                | Training Set   | Test Set        | External Validation | Temporal Validation |
| --------------------- | -------------- | --------------- | ------------------- | ------------------- |
| **AUC**         | 0.788 ± 0.083 | **0.828** | 0.856               | 0.843               |
| **Sensitivity** | -              | **92.7%** | 90.0%               | 88.0%               |
| **Specificity** | -              | **65.2%** | 70.0%               | 68.0%               |
| **Accuracy**    | -              | **81.5%** | 82.0%               | 80.0%               |

### Optimal Classification Threshold

- **Threshold**: 0.4963 (determined by Youden's index)

## 🔬 Final 7 Features

After SHAP importance analysis and DeLong test, the final model uses the following 7 features:

1. **preop_crp** - Preoperative C-reactive protein (mg/L) ⭐ Most important
2. **MLR** - Monocyte-to-lymphocyte ratio ⭐ Second most important
3. **NLR** - Neutrophil-to-lymphocyte ratio
4. **diameter** - Appendiceal diameter (mm) ⭐ Important
5. **weight** - Body weight (kg)
6. **preop_plt** - Preoperative platelet count (×10⁹/L)
7. **NMLR** - Neutrophil/(monocyte+lymphocyte) ratio

> **Note**: MLR, NLR, and NMLR are derived indicators, automatically calculated from basic laboratory values


### Data Preprocessing

- MICE multiple imputation for missing values
- Excluded features with >80% missing rate (preop_pct)
- Impute basic indicators first, then calculate derived indicators
- Ensure mathematical logic consistency

## 🚀 Quick Start

### 1. Requirements

```bash
Python >= 3.8
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

- pandas >= 1.3.0
- numpy >= 1.21.0, <2.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- joblib >= 1.0.0

### 3. Run Analysis

#### Data Preprocessing

```bash
# Open and run
jupyter notebook 阑尾炎数据分析.ipynb
```

#### Model Training

```bash
# Open and run
jupyter notebook 阑尾炎机器学习_ECM方法.ipynb
```

#### External Validation

```bash
# Open and run
jupyter notebook 阑尾炎机器学习_外部验证与时序验证.ipynb
```

## 🌐 Web Application Deployment

This project includes a complete Streamlit Web application that can be deployed to Streamlit Community Cloud.

### Local Run

```bash
cd web
python setup.py  # Copy model files
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Push code to a public GitHub repository
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/)
3. Login with your GitHub account
4. Click "New app" and configure:
   - **Repository**: Your GitHub repository
   - **Branch**: main
   - **Main file path**: `web/app.py`
5. Click "Deploy"

For detailed deployment guide, see [web/DEPLOY.md](web/DEPLOY.md)

## 🔬 Research Methods

### Model Selection

- Compared 11 machine learning algorithms (RF, GBM, XGBoost, AdaBoost, SVM, LR, etc.)
- Evaluated using 10-fold stratified cross-validation
- Selected best model based on test set AUC

### Feature Selection

- Used SHAP values to assess feature importance
- Gradually reduced features through DeLong test
- Finally determined 7 optimal features

### Model Validation

- **Internal Validation**: 70% training / 30% test
- **External Validation**: Different hospital data
- **Temporal Validation**: Different time period data

### Performance Evaluation

- ROC curve and AUC
- Sensitivity, Specificity, PPV, NPV
- Confusion matrix
- Calibration curve
- Decision curve analysis (DCA)

## 📈 Main Results

### Model Comparison (Test Set AUC)

| Model                 | AUC             | Sensitivity     | Specificity     |
| --------------------- | --------------- | --------------- | --------------- |
| **AdaBoost** ⭐ | **0.828** | **92.7%** | **65.2%** |
| SVM                   | 0.826           | 91.7%           | 60.6%           |
| XGBoost               | 0.825           | 89.6%           | 65.2%           |
| Random Forest         | 0.811           | 91.7%           | 56.1%           |
| Logistic Regression   | 0.798           | 88.5%           | 62.1%           |

### Feature Importance (SHAP Values)

1. **preop_crp** (0.597) - Preoperative C-reactive protein
2. **MLR** (0.320) - Monocyte-to-lymphocyte ratio
3. **diameter** (0.215) - Appendiceal diameter
4. **weight** (0.110) - Body weight
5. **preop_lymph** (0.103) - Preoperative lymphocyte count

## 📝 Usage Instructions


### For Clinicians

1. Use the Web application for prediction (recommended)
2. Or directly load model files for batch prediction

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('结果/final_model.pkl')

# Prepare data (7 features)
features = ['preop_crp', 'MLR', 'NLR', 'diameter', 'weight', 'preop_plt', 'NMLR']
X = df[features]

# Predict
prob = model.predict_proba(X)[:, 1]
pred = (prob >= 0.4963).astype(int)
```

## ⚠️ Important Notes

1. **Data Privacy**: Data used in this project has been de-identified and does not contain patient privacy information
2. **Clinical Use**: This model is for reference only and cannot replace professional medical diagnosis
3. **Model Limitations**: The model was trained on specific population data and may require re-validation in different populations
4. **Feature Requirements**: Ensure all 7 input features are complete and accurate

## 📚 Related Documentation

- [Web Application Deployment Guide](web/DEPLOY.md)
- [Web Application Quick Start](web/QUICKSTART.md)
- [Complete Project Summary](总结/项目完整总结.md)
- [External Validation Report](总结/外部验证报告.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit Issues and Pull Requests.


## 📄 License

This project is for research use only. For commercial use, please contact the author.

## ⚠️ Disclaimer

This prediction system is for reference only and cannot replace professional medical diagnosis. All medical decisions should be made by professional doctors. The author is not responsible for any consequences arising from the use of this system.

## 📧 Contact

For questions or suggestions, please contact:
- Email: xdk1207@sina.com

## 🙏 Acknowledgments

Thanks to all clinicians and researchers who participated in data collection and model validation.

---

**Version**: v1.0
**Last Updated**: 2026
**Maintenance Status**: ✅ Actively maintained
