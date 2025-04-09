import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load("titanic-app/titanic_rf_model.pkl")
feature_columns = joblib.load("titanic-app/feature_columns.pkl")

def preprocess_input(input_data):
    # Crear DataFrame base
    df = pd.DataFrame([input_data])
    
    # ========== TRANSFORMACIONES MANUALES ==========
    # 1. Codificación directa de variables categóricas
    title_mapping = {
        "Mr": "Title_Mr",
        "Mrs": "Title_Mrs", 
        "Miss": "Title_Miss",
        "Master": "Title_Master",
        "Rare": "Title_Rare"
    }
    df[title_mapping.values()] = 0
    df[title_mapping[input_data["Title"]]] = 1
    
    # 2. Codificación de AgeGroup
    age = input_data["Age"]
    age_groups = {
        "AgeGroup_Niño": (0 <= age < 12),
        "AgeGroup_Adolescente": (12 <= age < 18),
        "AgeGroup_Adulto Joven": (18 <= age < 35),
        "AgeGroup_Adulto": (35 <= age < 60),
        "AgeGroup_Adulto Mayor": (age >= 60)
    }
    for col, condition in age_groups.items():
        df[col] = 1 if condition else 0
    
    # 3. Variables calculadas
    df["FamilySize"] = df["SibSp_clean"] + df["Parch_clean"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    
    # 4. Asegurar todas las columnas esperadas
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
            
    return df[feature_columns]

st.title("🚢 Simulador de Supervivencia - Titanic")
st.markdown("### ¿Habrías sobrevivido al Titanic? Ingresa tus datos:")

with st.form("prediccion_form"):
    st.subheader("1. Información Personal")
    
    col1, col2 = st.columns(2)
    with col1:
        estrato = st.selectbox("Estrato socioeconómico (Colombia)", [1,2,3,4,5,6])
        pclass = 3 if estrato <=2 else (2 if estrato <=4 else 1)
        
    with col2:
        sex = st.radio("Género", ["female", "male"], format_func=lambda x: "Mujer 👩" if x == "female" else "Hombre 👨")
        sex_male = 1 if sex == "male" else 0
    
    # ========== SECCIÓN 2: CONTEXTO DEL VIAJE ==========
    st.subheader("2. Contexto del Viaje")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        embarked = st.selectbox("Puerto de Embarque", ["C", "Q", "S"], 
                              format_func=lambda x: {"C": "Cherburgo 🇫🇷", "Q": "Queenstown 🇮🇪", "S": "Southampton 🇬🇧"}[x])
        embarked_q = 1 if embarked == "Q" else 0
        embarked_s = 1 if embarked == "S" else 0
        
    with col2:
        fare = st.slider("Tarifa pagada (USD)", 0.0, 600.0, 32.0)
        fare_log = np.log1p(fare)
        
    with col3:
        age = st.number_input("Edad", min_value=0, max_value=100, value=25)
    
    # ========== SECCIÓN 3: FAMILIA ==========
    st.subheader("3. Familia a Bordo")
    
    col1, col2 = st.columns(2)
    with col1:
        sibsp = st.number_input("Hermanos/cónyuge", 0, 10, 0)
        sibsp_clean = min(sibsp, 3)
        
    with col2:
        parch = st.number_input("Padres/hijos", 0, 10, 0)
        parch_clean = min(parch, 2)
    
    # ========== SECCIÓN 4: TÍTULO SOCIAL ==========
    st.subheader("4. Título Social")
    
    title_options = {
        "male": ["Mr", "Master", "Rare"],
        "female": ["Miss", "Mrs", "Rare"]
    }
    title = st.selectbox("Selecciona tu título", 
                        title_options[sex],
                        format_func=lambda x: {
                            "Mr": "Señor (Adulto)",
                            "Master": "Master (Niño/Joven)",
                            "Miss": "Señorita (Soltera)",
                            "Mrs": "Señora (Casada)", 
                            "Rare": "Título Especial"
                        }[x])
    
    submitted = st.form_submit_button("🔮 Predecir Supervivencia")

if submitted:
    input_data = {
        "Pclass": pclass,
        "Sex_male": sex_male,
        "Embarked_Q": embarked_q,
        "Embarked_S": embarked_s,
        "Fare_log": fare_log,
        "SibSp_clean": sibsp_clean,
        "Parch_clean": parch_clean,
        "Title": title,
        "Age": age
    }
    
    processed_data = preprocess_input(input_data)
    
    try:
        prediction = model.predict(processed_data)[0]
        proba = model.predict_proba(processed_data)[0][1] * 100
        
        st.divider()
        if prediction == 1:
            st.success(f"## ✅ SOBREVIVIRÍAS ({proba:.1f}% de probabilidad)")
            st.balloons()
        else:
            st.error(f"## ❌ NO SOBREVIVIRÍAS ({proba:.1f}% de probabilidad)")
            
        with st.expander("📊 Detalles técnicos"):
            st.write("**Variables utilizadas:**")
            st.dataframe(processed_data.T.style.format("{:.1f}"))
            
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        st.write("Columnas generadas:", processed_data.columns.tolist())
        st.write("Columnas esperadas:", feature_columns)
st.sidebar.markdown(
    """
    <a href="https://www.linkedin.com/in/andres-felipe-lemus-v-7943882a9/" target="_blank" style="text-decoration:none;">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20" style="vertical-align:middle; margin-right:5px;" />
        Andres Felipe Lemus
    </a>
    """,
    unsafe_allow_html=True
)


st.markdown("---")
st.caption("Modelo desarrollado con datos históricos del Titanic - Fines demostrativos")