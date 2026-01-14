"""
阑尾炎复杂性预测Web应用
基于AdaBoost机器学习模型（7个特征）
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# 页面配置
st.set_page_config(
    page_title="阑尾炎复杂性预测系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E64B35;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #E64B35;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffe6e6;
        border-left-color: #ff0000;
    }
    .low-risk {
        background-color: #e6f3ff;
        border-left-color: #0066cc;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== 最终使用的7个特征 =====
FINAL_FEATURES = ['preop_crp', 'MLR', 'NLR', 'diameter', 'weight', 'preop_plt', 'NMLR']

# 最佳分类阈值（从训练时确定）
OPTIMAL_THRESHOLD = 0.4963

# 特征中文名称映射（仅7个特征）
FEATURE_NAMES_CN = {
    'preop_crp': '术前CRP（mg/L）',
    'MLR': 'MLR（单核细胞/淋巴细胞比值）',
    'NLR': 'NLR（中性粒细胞/淋巴细胞比值）',
    'diameter': '阑尾直径（mm）',
    'weight': '体重（kg）',
    'preop_plt': '术前血小板（×10⁹/L）',
    'NMLR': 'NMLR（中性粒细胞/单核细胞+淋巴细胞比值）'
}

# 特征单位/说明
FEATURE_UNITS = {
    'preop_crp': 'mg/L',
    'MLR': '比值（自动计算）',
    'NLR': '比值（自动计算）',
    'diameter': 'mm',
    'weight': 'kg',
    'preop_plt': '×10⁹/L',
    'NMLR': '比值（自动计算）'
}

# 基础特征（用于计算衍生指标）
BASE_FEATURES = {
    'preop_neut': '术前中性粒细胞（×10⁹/L）',
    'preop_lymph': '术前淋巴细胞（×10⁹/L）',
    'preop_mono': '术前单核细胞（×10⁹/L）',
    'preop_wbc': '术前WBC（×10⁹/L）'
}

@st.cache_resource
def load_model():
    """加载训练好的模型"""
    try:
        # 尝试从不同路径加载模型
        model_paths = [
            '../结果/final_model.pkl',
            '结果/final_model.pkl',
            './final_model.pkl',
            '../结果/model_AdaBoost.pkl',  # 备用路径
            '结果/model_AdaBoost.pkl',
            './model_AdaBoost.pkl'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                st.success(f"✅ 成功加载模型: {path}")
                return model
        
        st.error("❌ 未找到模型文件！请确保模型文件在正确的位置。")
        st.info("💡 请运行 setup.py 或手动复制 final_model.pkl 到 web/ 目录")
        return None
    except Exception as e:
        st.error(f"❌ 加载模型时出错: {str(e)}")
        return None

def calculate_derived_features(input_data):
    """计算衍生特征（MLR, NLR, NMLR）"""
    df = pd.DataFrame([input_data])
    
    # 计算NLR（中性粒细胞/淋巴细胞）
    if 'preop_neut' in df.columns and 'preop_lymph' in df.columns:
        df['NLR'] = df['preop_neut'] / (df['preop_lymph'] + 1e-10)
    else:
        df['NLR'] = np.nan
    
    # 计算MLR（单核细胞/淋巴细胞）
    if 'preop_mono' in df.columns and 'preop_lymph' in df.columns:
        df['MLR'] = df['preop_mono'] / (df['preop_lymph'] + 1e-10)
    else:
        df['MLR'] = np.nan
    
    # 计算NMLR（中性粒细胞/(单核细胞+淋巴细胞)）
    if 'preop_neut' in df.columns and 'preop_mono' in df.columns and 'preop_lymph' in df.columns:
        df['NMLR'] = df['preop_neut'] / (df['preop_mono'] + df['preop_lymph'] + 1e-10)
    else:
        df['NMLR'] = np.nan
    
    return df.iloc[0].to_dict()

def predict_risk(model, input_data):
    """进行预测（不使用标准化）"""
    try:
        # 准备输入数据
        X = pd.DataFrame([input_data])
        
        # 确保所有必需特征都存在
        missing_features = [f for f in FINAL_FEATURES if f not in X.columns]
        if missing_features:
            st.error(f"❌ 缺少以下必需特征: {', '.join(missing_features)}")
            return None
        
        # 只选择模型需要的7个特征
        X = X[FINAL_FEATURES]
        
        # 检查是否有缺失值
        if X.isnull().any().any():
            missing_cols = X.columns[X.isnull().any()].tolist()
            st.error(f"❌ 以下特征有缺失值: {', '.join(missing_cols)}")
            return None
        
        # 预测概率（AdaBoost不需要标准化）
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)[0, 1]
        else:
            prob = model.predict(X)[0]
        
        return prob
    except Exception as e:
        st.error(f"❌ 预测时出错: {str(e)}")
        st.exception(e)
        return None

def main():
    # 标题
    st.markdown('<div class="main-header">🏥 阑尾炎复杂性预测系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">基于AdaBoost机器学习模型 | 7个特征 | 测试集AUC = 0.828</div>', unsafe_allow_html=True)
    
    # 加载模型
    model = load_model()
    if model is None:
        st.stop()
    
    # 侧边栏说明
    with st.sidebar:
        st.header("📋 使用说明")
        st.markdown("""
        **功能说明：**
        - 输入患者术前临床特征
        - 系统自动计算衍生比值指标
        - 预测复杂阑尾炎风险概率
        
        **模型信息：**
        - 算法：AdaBoost
        - 特征数量：7个
        - 测试集AUC：0.828
        - 最佳阈值：0.4963
        
        **7个特征：**
        1. 术前CRP
        2. MLR（自动计算）
        3. NLR（自动计算）
        4. 阑尾直径
        5. 体重
        6. 术前血小板
        7. NMLR（自动计算）
        
        **注意事项：**
        - 所有输入均为术前可获得的数据
        - 系统会自动计算MLR、NLR、NMLR
        - 预测结果仅供参考，需结合临床判断
        """)
        
        st.markdown("---")
        st.markdown("**📊 模型性能指标**")
        st.metric("AUC", "0.828")
        st.metric("敏感性", "92.7%")
        st.metric("特异性", "65.2%")
        st.metric("准确率", "81.5%")
        st.metric("最佳阈值", "0.4963")
    
    # 主界面：输入表单
    st.header("📝 患者信息输入")
    st.info("💡 **提示**：请填写以下信息，系统会自动计算MLR、NLR、NMLR等衍生指标")
    
    # 使用列布局
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        st.subheader("基础检验指标（用于计算衍生指标）")
        input_data['preop_neut'] = st.number_input(
            BASE_FEATURES['preop_neut'],
            min_value=0.0,
            max_value=30.0,
            value=7.0,
            step=0.1,
            help="用于计算NLR和NMLR"
        )
        input_data['preop_lymph'] = st.number_input(
            BASE_FEATURES['preop_lymph'],
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="用于计算MLR、NLR和NMLR"
        )
        input_data['preop_mono'] = st.number_input(
            BASE_FEATURES['preop_mono'],
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="用于计算MLR和NMLR"
        )
        input_data['preop_wbc'] = st.number_input(
            BASE_FEATURES['preop_wbc'],
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=0.1,
            help="白细胞计数"
        )
    
    with col2:
        st.subheader("模型所需特征")
        input_data['preop_crp'] = st.number_input(
            FEATURE_NAMES_CN['preop_crp'],
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=1.0,
            help=FEATURE_UNITS['preop_crp']
        )
        input_data['diameter'] = st.number_input(
            FEATURE_NAMES_CN['diameter'],
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=0.1,
            help=FEATURE_UNITS['diameter']
        )
        input_data['weight'] = st.number_input(
            FEATURE_NAMES_CN['weight'],
            min_value=10.0,
            max_value=200.0,
            value=70.0,
            step=1.0,
            help=FEATURE_UNITS['weight']
        )
        input_data['preop_plt'] = st.number_input(
            FEATURE_NAMES_CN['preop_plt'],
            min_value=0.0,
            max_value=1000.0,
            value=250.0,
            step=10.0,
            help=FEATURE_UNITS['preop_plt']
        )
    
    # 计算衍生特征
    input_data = calculate_derived_features(input_data)
    
    # 显示计算的衍生指标
    with st.expander("📈 查看自动计算的衍生指标", expanded=True):
        col_der1, col_der2, col_der3 = st.columns(3)
        with col_der1:
            st.metric("NLR", f"{input_data.get('NLR', 0):.3f}", 
                     help="中性粒细胞/淋巴细胞比值")
        with col_der2:
            st.metric("MLR", f"{input_data.get('MLR', 0):.3f}",
                     help="单核细胞/淋巴细胞比值")
        with col_der3:
            st.metric("NMLR", f"{input_data.get('NMLR', 0):.3f}",
                     help="中性粒细胞/(单核细胞+淋巴细胞)比值")
    
    # 预测按钮
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 开始预测", type="primary", use_container_width=True)
    
    # 显示预测结果
    if predict_button:
        # 验证必需特征是否完整
        missing = [f for f in FINAL_FEATURES if f not in input_data or pd.isna(input_data.get(f))]
        if missing:
            st.error(f"❌ 以下特征缺失或无效: {', '.join(missing)}")
            st.info("💡 请确保所有基础检验指标都已填写，系统会自动计算衍生指标")
        else:
            with st.spinner("正在计算预测结果..."):
                prob = predict_risk(model, input_data)
            
            if prob is not None:
                # 风险等级判断（使用最佳阈值0.4963）
                risk_level = "高风险" if prob >= OPTIMAL_THRESHOLD else "低风险"
                risk_class = "high-risk" if prob >= OPTIMAL_THRESHOLD else "low-risk"
                
                # 显示预测结果
                st.markdown("---")
                st.header("📊 预测结果")
                
                # 主要预测框
                if prob >= OPTIMAL_THRESHOLD:
                    st.markdown(f"""
                    <div class="prediction-box high-risk">
                        <h2 style="color: #ff0000; margin-bottom: 0.5rem;">⚠️ 高风险：复杂阑尾炎</h2>
                        <h1 style="color: #ff0000; font-size: 3rem; margin: 0;">{prob:.1%}</h1>
                        <p style="margin-top: 0.5rem; color: #666;">建议：密切观察，考虑早期手术干预</p>
                        <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #999;">阈值：{OPTIMAL_THRESHOLD:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box low-risk">
                        <h2 style="color: #0066cc; margin-bottom: 0.5rem;">✓ 低风险：单纯阑尾炎</h2>
                        <h1 style="color: #0066cc; font-size: 3rem; margin: 0;">{prob:.1%}</h1>
                        <p style="margin-top: 0.5rem; color: #666;">建议：常规治疗，继续观察</p>
                        <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #999;">阈值：{OPTIMAL_THRESHOLD:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 详细指标
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                with col_met1:
                    st.metric("预测概率", f"{prob:.1%}")
                with col_met2:
                    st.metric("风险等级", risk_level)
                with col_met3:
                    st.metric("分类阈值", f"{OPTIMAL_THRESHOLD:.1%}")
                with col_met4:
                    st.metric("模型AUC", "0.828")
                
                # 显示所有输入数据（仅7个特征）
                with st.expander("📋 查看模型使用的7个特征值"):
                    feature_data = {FEATURE_NAMES_CN[f]: input_data[f] for f in FINAL_FEATURES if f in input_data}
                    df_features = pd.DataFrame([feature_data])
                    st.dataframe(df_features, use_container_width=True)
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>⚠️ <strong>免责声明</strong>：本预测系统仅供参考，不能替代专业医疗诊断。所有医疗决策应由专业医生做出。</p>
        <p>基于AdaBoost机器学习模型 | 7个特征 | 测试集AUC = 0.828 | 最佳阈值 = 0.4963</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
