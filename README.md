# Análisis Predictivo de Supervivencia en el Titanic 🚢
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://titanic-tsby9nt8wvdhukq4ppnmdz.streamlit.app/)

Este proyecto combina análisis exploratorio de datos (EDA) y machine learning para predecir la supervivencia de pasajeros del Titanic, revelando patrones sociohistóricos clave mediante ciencia de datos.

## Objetivo General

Desarrollar un modelo predictivo que identifique los factores determinantes en la supervivencia durante el hundimiento del Titanic, con énfasis en:
- Relaciones entre variables demográficas y supervivencia
- Impacto de características socioeconómicas
- Interpretabilidad del modelo para audiencias técnicas y no técnicas

## Características Principales

### 1. Análisis Exploratorio (EDA) Avanzado
- Visualización de patrones de supervivencia por género, clase y edad
- Manejo estratégico de valores faltantes (77% en `Cabin`)
- Ingeniería de variables contextuales:
  - `FamilySize`: Tamaño del grupo familiar
  - `IsAlone`: Viajeros solitarios
  - `Title`: Extracción de títulos sociales desde nombres

### 2. Pipeline de Machine Learning
- **Preprocesamiento robusto**:
  - Transformación logarítmica para `Fare`
  - Codificación manual de variables categóricas
  - Tratamiento de outliers en `SibSp` y `Parch`
  
- **Comparación de modelos**:
  - Regresión Logística (baseline)
  - Random Forest (modelo final)
  - XGBoost (alto rendimiento)

- **Optimización de hiperparámetros**:
  - Búsqueda en grilla con validación cruzada
  - Métrica principal: AUC-ROC (0.90 en mejor modelo)

### 3. Dashboard Interactivo
- **Simulador de supervivencia**:
  - Inputs contextualizados (ej: estratos colombianos → clases del Titanic)
  - Visualización de probabilidades con explicaciones técnicas
  - Detección automática de patrones clave en predicciones

- **Visualizaciones profesionales**:
  - Matriz de confusión interactiva
  - Importancia de variables
  - Curvas ROC comparativas

## Estructura del Proyecto
titanic-analysis/
├── data/ # Datasets originales
├── notebooks/ # Jupyter notebooks de análisis
├── models/ # Modelos serializados (.pkl)
├── app/ # Código de la aplicación Streamlit
│ ├── app.py # Lógica principal
│ └── assets/ # Imágenes y recursos visuales
└── requirements.txt # Dependencias


## Guía de Uso
1. **Instalación**:
```bash
git clone https://github.com/Lemus2901/Titanic.git
pip install -r requirements.txt
