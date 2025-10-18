import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🧬",
    layout="wide"
)

# --- FUNÇÃO DE CACHE PARA CARREGAR O MODELO E O SCALER ---
# O cache garante que o modelo e o scaler sejam carregados apenas uma vez, otimizando a performance.
@st.cache_data
def load_model_and_scaler():
    with open('breast_cancer_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

model, scaler = load_model_and_scaler()

# --- SIDEBAR COM OS INPUTS DO USUÁRIO ---
st.sidebar.header('Input Cell Nucleus Measurements')

def add_sidebar_inputs():
    # Esta função cria sliders na sidebar para cada uma das 30 features.
    # Os valores de min, max e default foram baseados na descrição estatística do dataset original (df.describe())
    # Isso torna a experiência do usuário mais realista.
    
    # Criamos um dicionário para armazenar os inputs
    data = {
        'radius_mean': st.sidebar.slider('Radius (mean)', 6.9, 28.2, 14.1),
        'texture_mean': st.sidebar.slider('Texture (mean)', 9.7, 39.3, 19.2),
        'perimeter_mean': st.sidebar.slider('Perimeter (mean)', 43.7, 188.5, 91.9),
        'area_mean': st.sidebar.slider('Area (mean)', 143.5, 2501.0, 654.8),
        'smoothness_mean': st.sidebar.slider('Smoothness (mean)', 0.05, 0.16, 0.09),
        'compactness_mean': st.sidebar.slider('Compactness (mean)', 0.01, 0.35, 0.1),
        'concavity_mean': st.sidebar.slider('Concavity (mean)', 0.0, 0.43, 0.08),
        'concave points_mean': st.sidebar.slider('Concave Points (mean)', 0.0, 0.2, 0.04),
        'symmetry_mean': st.sidebar.slider('Symmetry (mean)', 0.1, 0.3, 0.18),
        'fractal_dimension_mean': st.sidebar.slider('Fractal Dimension (mean)', 0.04, 0.09, 0.06),
        
        'radius_se': st.sidebar.slider('Radius (se)', 0.11, 2.87, 0.4),
        'texture_se': st.sidebar.slider('Texture (se)', 0.36, 4.88, 1.21),
        'perimeter_se': st.sidebar.slider('Perimeter (se)', 0.75, 21.98, 2.86),
        'area_se': st.sidebar.slider('Area (se)', 6.8, 542.2, 40.33),
        'smoothness_se': st.sidebar.slider('Smoothness (se)', 0.001, 0.031, 0.007),
        'compactness_se': st.sidebar.slider('Compactness (se)', 0.002, 0.135, 0.025),
        'concavity_se': st.sidebar.slider('Concavity (se)', 0.0, 0.39, 0.031),
        'concave points_se': st.sidebar.slider('Concave Points (se)', 0.0, 0.052, 0.011),
        'symmetry_se': st.sidebar.slider('Symmetry (se)', 0.007, 0.078, 0.02),
        'fractal_dimension_se': st.sidebar.slider('Fractal Dimension (se)', 0.0008, 0.029, 0.003),
        
        'radius_worst': st.sidebar.slider('Radius (worst)', 7.9, 36.0, 16.2),
        'texture_worst': st.sidebar.slider('Texture (worst)', 12.0, 49.5, 25.6),
        'perimeter_worst': st.sidebar.slider('Perimeter (worst)', 50.4, 251.2, 107.2),
        'area_worst': st.sidebar.slider('Area (worst)', 185.2, 4254.0, 880.5),
        'smoothness_worst': st.sidebar.slider('Smoothness (worst)', 0.07, 0.22, 0.13),
        'compactness_worst': st.sidebar.slider('Compactness (worst)', 0.02, 1.05, 0.25),
        'concavity_worst': st.sidebar.slider('Concavity (worst)', 0.0, 1.25, 0.27),
        'concave points_worst': st.sidebar.slider('Concave Points (worst)', 0.0, 0.29, 0.11),
        'symmetry_worst': st.sidebar.slider('Symmetry (worst)', 0.15, 0.66, 0.29),
        'fractal_dimension_worst': st.sidebar.slider('Fractal Dimension (worst)', 0.05, 0.2, 0.08)
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = add_sidebar_inputs()

# --- PÁGINA PRINCIPAL ---

# Título e Introdução
st.write("# 🧬 Breast Cancer Predictor")
st.write("Esta aplicação utiliza um modelo de Machine Learning para prever se um tumor de mama é Benigno ou Maligno com base em medições de exames. Use a barra lateral para inserir os dados.")

st.markdown("---")

# Mostrar os dados de entrada do usuário
st.subheader('User Input Features')
st.write(input_df)

# --- PREDIÇÃO E RESULTADOS ---

if st.sidebar.button('Predict'):
    # Escalar os dados de entrada
    input_scaled = scaler.transform(input_df)
    
    # Fazer a predição
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    st.subheader('Prediction Result')
    
    # Mostrar o resultado
    if prediction[0] == 0:
        st.success('**Diagnosis: Benign**')
        st.write(f"**Confidence:** {prediction_proba[0][0]*100:.2f}%")
    else:
        st.error('**Diagnosis: Malignant**')
        st.write(f"**Confidence:** {prediction_proba[0][1]*100:.2f}%")

st.markdown("---")
# Aviso Legal
st.warning("**Disclaimer:** This is a tool for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")
