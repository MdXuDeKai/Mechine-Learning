"""
Preoperative Prediction Web Application for Complicated Appendicitis
Based on AdaBoost Machine Learning Model (7 Features)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Complicated Appendicitis Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
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

# ===== Final 7 Features =====
FINAL_FEATURES = ['preop_crp', 'MLR', 'NLR', 'diameter', 'weight', 'preop_plt', 'NMLR']

# Optimal classification threshold (determined during training)
OPTIMAL_THRESHOLD = 0.4963

# Feature English names (7 features only)
FEATURE_NAMES_EN = {
    'preop_crp': 'Preoperative CRP (mg/L)',
    'MLR': 'MLR (Monocyte-to-Lymphocyte Ratio)',
    'NLR': 'NLR (Neutrophil-to-Lymphocyte Ratio)',
    'diameter': 'Appendiceal Diameter (mm)',
    'weight': 'Body Weight (kg)',
    'preop_plt': 'Preoperative Platelet Count (×10⁹/L)',
    'NMLR': 'NMLR (Neutrophil/(Monocyte+Lymphocyte) Ratio)'
}

# Feature units/descriptions
FEATURE_UNITS = {
    'preop_crp': 'mg/L',
    'MLR': 'Ratio (auto-calculated)',
    'NLR': 'Ratio (auto-calculated)',
    'diameter': 'mm',
    'weight': 'kg',
    'preop_plt': '×10⁹/L',
    'NMLR': 'Ratio (auto-calculated)'
}

# Base features (for calculating derived indicators)
BASE_FEATURES = {
    'preop_neut': 'Preoperative Neutrophil Count (×10⁹/L)',
    'preop_lymph': 'Preoperative Lymphocyte Count (×10⁹/L)',
    'preop_mono': 'Preoperative Monocyte Count (×10⁹/L)',
    'preop_wbc': 'Preoperative WBC Count (×10⁹/L)'
}

@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        # Get current directory
        current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
        
        # Try multiple paths (for both local and Streamlit Cloud)
        model_paths = [
            # Current directory (web/)
            current_dir / 'final_model.pkl',
            # Parent directory
            current_dir.parent / '结果' / 'final_model.pkl',
            current_dir.parent / 'results' / 'final_model.pkl',
            # Relative paths
            Path('final_model.pkl'),
            Path('./final_model.pkl'),
            Path('../结果/final_model.pkl'),
            Path('../results/final_model.pkl'),
            Path('结果/final_model.pkl'),
            Path('results/final_model.pkl'),
            # Backup paths
            current_dir / 'model_AdaBoost.pkl',
            current_dir.parent / '结果' / 'model_AdaBoost.pkl',
            Path('model_AdaBoost.pkl'),
        ]
        
        for path in model_paths:
            try:
                if path.exists():
                    model = joblib.load(str(path))
                    st.success(f"✅ Model loaded successfully: {path}")
                    return model
            except Exception as e:
                continue
        
        # If not found, show error with helpful message
        st.error("❌ Model file not found! Please ensure the model file is in the correct location.")
        st.info("💡 Please run setup.py or manually copy final_model.pkl to the web/ directory")
        st.info("💡 For Streamlit Cloud, ensure final_model.pkl is in the web/ directory in your GitHub repository")
        
        # Show current directory for debugging
        with st.expander("🔍 Debug Information"):
            st.write(f"Current directory: {current_dir}")
            st.write(f"Current directory exists: {current_dir.exists()}")
            st.write(f"Files in current directory:")
            try:
                files = list(current_dir.glob('*.pkl'))
                if files:
                    for f in files:
                        st.write(f"  - {f}")
                else:
                    st.write("  No .pkl files found")
            except:
                st.write("  Cannot list files")
        
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.exception(e)
        return None

def calculate_derived_features(input_data):
    """Calculate derived features (MLR, NLR, NMLR)"""
    df = pd.DataFrame([input_data])
    
    # Calculate NLR (Neutrophil/Lymphocyte)
    if 'preop_neut' in df.columns and 'preop_lymph' in df.columns:
        df['NLR'] = df['preop_neut'] / (df['preop_lymph'] + 1e-10)
    else:
        df['NLR'] = np.nan
    
    # Calculate MLR (Monocyte/Lymphocyte)
    if 'preop_mono' in df.columns and 'preop_lymph' in df.columns:
        df['MLR'] = df['preop_mono'] / (df['preop_lymph'] + 1e-10)
    else:
        df['MLR'] = np.nan
    
    # Calculate NMLR (Neutrophil/(Monocyte+Lymphocyte))
    if 'preop_neut' in df.columns and 'preop_mono' in df.columns and 'preop_lymph' in df.columns:
        df['NMLR'] = df['preop_neut'] / (df['preop_mono'] + df['preop_lymph'] + 1e-10)
    else:
        df['NMLR'] = np.nan
    
    return df.iloc[0].to_dict()

def predict_risk(model, input_data):
    """Perform prediction (no standardization needed)"""
    try:
        # Prepare input data
        X = pd.DataFrame([input_data])
        
        # Ensure all required features exist
        missing_features = [f for f in FINAL_FEATURES if f not in X.columns]
        if missing_features:
            st.error(f"❌ Missing required features: {', '.join(missing_features)}")
            return None
        
        # Select only the 7 features needed by the model
        X = X[FINAL_FEATURES]
        
        # Check for missing values
        if X.isnull().any().any():
            missing_cols = X.columns[X.isnull().any()].tolist()
            st.error(f"❌ Missing values in features: {', '.join(missing_cols)}")
            return None
        
        # Predict probability (AdaBoost doesn't need standardization)
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)[0, 1]
        else:
            prob = model.predict(X)[0]
        
        return prob
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.exception(e)
        return None

def main():
    # Title
    st.markdown('<div class="main-header">🏥 Complicated Appendicitis Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Based on AdaBoost Machine Learning Model | 7 Features | Test AUC = 0.828</div>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        **Function:**
        - Input patient preoperative clinical features
        - System automatically calculates derived ratio indicators
        - Predict complicated appendicitis risk probability
        
        **Model Information:**
        - Algorithm: AdaBoost
        - Number of Features: 7
        - Test Set AUC: 0.828
        - Optimal Threshold: 0.4963
        
        **7 Features:**
        1. Preoperative CRP
        2. MLR (auto-calculated)
        3. NLR (auto-calculated)
        4. Appendiceal diameter
        5. Body weight
        6. Preoperative platelet count
        7. NMLR (auto-calculated)
        
        **Notes:**
        - All inputs are preoperatively available data
        - System automatically calculates MLR, NLR, NMLR
        - Prediction results are for reference only, clinical judgment required
        """)
        
        st.markdown("---")
        st.markdown("**📊 Model Performance Metrics**")
        st.metric("AUC", "0.828")
        st.metric("Sensitivity", "92.7%")
        st.metric("Specificity", "65.2%")
        st.metric("Accuracy", "81.5%")
        st.metric("Optimal Threshold", "0.4963")
    
    # Main interface: Input form
    st.header("📝 Patient Information Input")
    st.info("💡 **Tip**: Please fill in the following information. The system will automatically calculate MLR, NLR, NMLR and other derived indicators.")
    
    # Use column layout
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        st.subheader("Basic Laboratory Values (for calculating derived indicators)")
        input_data['preop_neut'] = st.number_input(
            BASE_FEATURES['preop_neut'],
            min_value=0.0,
            max_value=30.0,
            value=7.0,
            step=0.1,
            help="Used to calculate NLR and NMLR"
        )
        input_data['preop_lymph'] = st.number_input(
            BASE_FEATURES['preop_lymph'],
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Used to calculate MLR, NLR and NMLR"
        )
        input_data['preop_mono'] = st.number_input(
            BASE_FEATURES['preop_mono'],
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="Used to calculate MLR and NMLR"
        )
        input_data['preop_wbc'] = st.number_input(
            BASE_FEATURES['preop_wbc'],
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=0.1,
            help="White blood cell count"
        )
    
    with col2:
        st.subheader("Model Required Features")
        input_data['preop_crp'] = st.number_input(
            FEATURE_NAMES_EN['preop_crp'],
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=1.0,
            help=FEATURE_UNITS['preop_crp']
        )
        input_data['diameter'] = st.number_input(
            FEATURE_NAMES_EN['diameter'],
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=0.1,
            help=FEATURE_UNITS['diameter']
        )
        input_data['weight'] = st.number_input(
            FEATURE_NAMES_EN['weight'],
            min_value=10.0,
            max_value=200.0,
            value=70.0,
            step=1.0,
            help=FEATURE_UNITS['weight']
        )
        input_data['preop_plt'] = st.number_input(
            FEATURE_NAMES_EN['preop_plt'],
            min_value=0.0,
            max_value=1000.0,
            value=250.0,
            step=10.0,
            help=FEATURE_UNITS['preop_plt']
        )
    
    # Calculate derived features
    input_data = calculate_derived_features(input_data)
    
    # Display calculated derived indicators
    with st.expander("📈 View Auto-Calculated Derived Indicators", expanded=True):
        col_der1, col_der2, col_der3 = st.columns(3)
        with col_der1:
            st.metric("NLR", f"{input_data.get('NLR', 0):.3f}", 
                     help="Neutrophil-to-Lymphocyte Ratio")
        with col_der2:
            st.metric("MLR", f"{input_data.get('MLR', 0):.3f}",
                     help="Monocyte-to-Lymphocyte Ratio")
        with col_der3:
            st.metric("NMLR", f"{input_data.get('NMLR', 0):.3f}",
                     help="Neutrophil/(Monocyte+Lymphocyte) Ratio")
    
    # Predict button
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Start Prediction", type="primary", use_container_width=True)
    
    # Display prediction results
    if predict_button:
        # Validate required features are complete
        missing = [f for f in FINAL_FEATURES if f not in input_data or pd.isna(input_data.get(f))]
        if missing:
            st.error(f"❌ Missing or invalid features: {', '.join(missing)}")
            st.info("💡 Please ensure all basic laboratory values are filled. The system will automatically calculate derived indicators.")
        else:
            with st.spinner("Calculating prediction results..."):
                prob = predict_risk(model, input_data)
            
            if prob is not None:
                # Risk level judgment (using optimal threshold 0.4963)
                risk_level = "High Risk" if prob >= OPTIMAL_THRESHOLD else "Low Risk"
                risk_class = "high-risk" if prob >= OPTIMAL_THRESHOLD else "low-risk"
                
                # Display prediction results
                st.markdown("---")
                st.header("📊 Prediction Results")
                
                # Main prediction box
                if prob >= OPTIMAL_THRESHOLD:
                    st.markdown(f"""
                    <div class="prediction-box high-risk">
                        <h2 style="color: #ff0000; margin-bottom: 0.5rem;">⚠️ High Risk: Complicated Appendicitis</h2>
                        <h1 style="color: #ff0000; font-size: 3rem; margin: 0;">{prob:.1%}</h1>
                        <p style="margin-top: 0.5rem; color: #666;">Recommendation: Close monitoring, consider early surgical intervention</p>
                        <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #999;">Threshold: {OPTIMAL_THRESHOLD:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box low-risk">
                        <h2 style="color: #0066cc; margin-bottom: 0.5rem;">✓ Low Risk: Simple Appendicitis</h2>
                        <h1 style="color: #0066cc; font-size: 3rem; margin: 0;">{prob:.1%}</h1>
                        <p style="margin-top: 0.5rem; color: #666;">Recommendation: Routine treatment, continue observation</p>
                        <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #999;">Threshold: {OPTIMAL_THRESHOLD:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed metrics
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                with col_met1:
                    st.metric("Predicted Probability", f"{prob:.1%}")
                with col_met2:
                    st.metric("Risk Level", risk_level)
                with col_met3:
                    st.metric("Classification Threshold", f"{OPTIMAL_THRESHOLD:.1%}")
                with col_met4:
                    st.metric("Model AUC", "0.828")
                
                # Display all input data (7 features only)
                with st.expander("📋 View 7 Features Used by Model"):
                    feature_data = {FEATURE_NAMES_EN[f]: input_data[f] for f in FINAL_FEATURES if f in input_data}
                    df_features = pd.DataFrame([feature_data])
                    st.dataframe(df_features, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>⚠️ <strong>Disclaimer</strong>: This prediction system is for reference only and cannot replace professional medical diagnosis. All medical decisions should be made by professional doctors.</p>
        <p>Based on AdaBoost Machine Learning Model | 7 Features | Test AUC = 0.828 | Optimal Threshold = 0.4963</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
