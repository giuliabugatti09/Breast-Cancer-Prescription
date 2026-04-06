import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Breast Cancer Predictor AI",
    page_icon="🧬",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL AND SCALER ---
@st.cache_resource
def load_assets():
    # Adjusting paths to look into the 'models' folder
    model_path = os.path.join('models', 'breast_cancer_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    
    # Fallback to root if not in models/
    if not os.path.exists(model_path):
        model_path = 'breast_cancer_model.pkl'
        scaler_path = 'scaler.pkl'
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header('🧬 Cell Measurements')
st.sidebar.write("Adjust the sliders to input cell nucleus data:")

def get_user_inputs():
    # Using Tabs in sidebar to organize 30 features
    tab_mean, tab_se, tab_worst = st.sidebar.tabs(["Mean", "SE", "Worst"])
    
    data = {}
    
    with tab_mean:
        data['radius_mean'] = st.slider('Radius', 6.9, 28.2, 14.1)
        data['texture_mean'] = st.slider('Texture', 9.7, 39.3, 19.2)
        data['perimeter_mean'] = st.slider('Perimeter', 43.7, 188.5, 91.9)
        data['area_mean'] = st.slider('Area', 143.5, 2501.0, 654.8)
        data['smoothness_mean'] = st.slider('Smoothness', 0.05, 0.16, 0.09)
        data['compactness_mean'] = st.slider('Compactness', 0.01, 0.35, 0.1)
        data['concavity_mean'] = st.slider('Concavity', 0.0, 0.43, 0.08)
        data['concave points_mean'] = st.slider('Concave Points', 0.0, 0.2, 0.04)
        data['symmetry_mean'] = st.slider('Symmetry', 0.1, 0.3, 0.18)
        data['fractal_dimension_mean'] = st.slider('Fractal Dim.', 0.04, 0.09, 0.06)

    with tab_se:
        data['radius_se'] = st.slider('Radius SE', 0.11, 2.87, 0.4)
        data['texture_se'] = st.slider('Texture SE', 0.36, 4.88, 1.21)
        data['perimeter_se'] = st.slider('Perimeter SE', 0.75, 21.98, 2.86)
        data['area_se'] = st.slider('Area SE', 6.8, 542.2, 40.33)
        data['smoothness_se'] = st.slider('Smoothness SE', 0.001, 0.031, 0.007)
        data['compactness_se'] = st.slider('Compactness SE', 0.002, 0.135, 0.025)
        data['concavity_se'] = st.slider('Concavity SE', 0.0, 0.39, 0.031)
        data['concave points_se'] = st.slider('Concave Points SE', 0.0, 0.052, 0.011)
        data['symmetry_se'] = st.slider('Symmetry SE', 0.007, 0.078, 0.02)
        data['fractal_dimension_se'] = st.slider('Fractal Dim. SE', 0.0008, 0.029, 0.003)

    with tab_worst:
        data['radius_worst'] = st.slider('Radius Worst', 7.9, 36.0, 16.2)
        data['texture_worst'] = st.slider('Texture Worst', 12.0, 49.5, 25.6)
        data['perimeter_worst'] = st.slider('Perimeter Worst', 50.4, 251.2, 107.2)
        data['area_worst'] = st.slider('Area Worst', 185.2, 4254.0, 880.5)
        data['smoothness_worst'] = st.slider('Smoothness Worst', 0.07, 0.22, 0.13)
        data['compactness_worst'] = st.slider('Compactness Worst', 0.02, 1.05, 0.25)
        data['concavity_worst'] = st.slider('Concavity Worst', 0.0, 1.25, 0.27)
        data['concave points_worst'] = st.slider('Concave Points Worst', 0.0, 0.29, 0.11)
        data['symmetry_worst'] = st.slider('Symmetry Worst', 0.15, 0.66, 0.29)
        data['fractal_dimension_worst'] = st.slider('Fractal Dim. Worst', 0.05, 0.2, 0.08)

    return pd.DataFrame(data, index=[0])

input_df = get_user_inputs()

# --- MAIN PAGE ---
st.title("🧬 Breast Cancer Diagnostic Assistant")
st.markdown("""
This application uses a **Logistic Regression** model trained on the *Wisconsin Diagnostic Breast Cancer* dataset. 
It analyzes 30 nuclear cell features to predict the probability of a tumor being **Benign** or **Malignant**.
""")

st.write("---")

col_input, col_result = st.columns([1.5, 1])

with col_input:
    st.subheader("📊 Current Input Parameters")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}), height=400)

with col_result:
    st.subheader("🎯 Prediction Result")
    
    if st.sidebar.button('Run Diagnostic Analysis 🚀'):
        # Scale inputs
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)
        proba = model.predict_proba(input_scaled)
        
        # Display Result
        if prediction[0] == 0:
            st.success("### Diagnosis: **BENIGN**")
            st.metric(label="Confidence Level", value=f"{proba[0][0]*100:.2f}%")
            st.write("The model suggests the tumor characteristics are consistent with non-cancerous samples.")
        else:
            st.error("### Diagnosis: **MALIGNANT**")
            st.metric(label="Confidence Level", value=f"{proba[0][1]*100:.2f}%")
            st.write("⚠️ **Attention:** The model identifies high-risk patterns consistent with malignancy.")
    else:
        st.info("Click the button on the sidebar to generate a prediction.")

st.write("---")
# Disclaimer in footer
st.caption("""
**Disclaimer:** This tool is for educational purposes and proof-of-concept only. 
It is powered by a Logistic Regression model (Accuracy: 97%). 
It should not be used as a medical diagnosis. Always consult with a healthcare professional.
""")