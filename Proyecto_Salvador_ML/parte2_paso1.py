# Tarea: Parte 2 - Implementación de modelos (Paso 1)
# Descripción: Entrenamiento de Árbol de Decisión y Random Forest para la 
#              predicción de enfermedades cardíacas.
# Gutierrez Hernandez Kevin Andrew
# Fecha: 19 de Marzo, 2026
# Fuente de datos: Heart Disease UCI Dataset (Kaggle)
# URL: https://www.kaggle.com/ronitf/heart-disease-uci


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report # Para evaluación detallada de métricas 

import os


# 1. Carga y exploración básica del dataset con pandas

base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, 'heart_disease.csv')
df = pd.read_csv(csv_path)

print("Primeras filas del dataset:")
print(df.head())

# Separación de características (X) y etiqueta (y)
X = df.drop('target', axis=1)
y = df['target']

# División del dataset en entrenamiento (80%) y prueba (20%)
# random_state=42 asegura que los resultados sean reproducibles.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. Entrenamiento del modelo: Árbol de Decisión (Decision Tree)

# DecisionTreeClassifier: Modelo que divide los datos basándose en reglas de decisión.
# Hiperparámetros:
# - criterion='gini': Mide la calidad de la división (impureza de Gini).
# - max_depth=5: Limita la profundidad para evitar el sobreajuste (overfitting).
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Predicciones
dt_preds = dt_model.predict(X_test)


# 3. Entrenamiento del modelo: Bosque Aleatorio (Random Forest)

# RandomForestClassifier: Conjunto de múltiples árboles de decisión (Ensemble).
# Hiperparámetros:
# - n_estimators=100: Número de árboles en el bosque. 100 es un valor estándar robusto.
# - max_depth=5: Para mantener consistencia con el modelo anterior y controlar complejidad.
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# Predicciones
rf_preds = rf_model.predict(X_test)


# 4. Evaluación de los modelos

# Se utilizan Accuracy (Exactitud) y F1-Score (balance entre precisión y exhaustividad).
print("\n--- Evaluación: Árbol de Decisión ---")
print(f"Accuracy: {accuracy_score(y_test, dt_preds):.4f}") #accuracy_score es una métrica que mide la proporción de predicciones correctas sobre el total de predicciones realizadas, proporcionando una visión general del desempeño del modelo.
print(f"F1-Score: {f1_score(y_test, dt_preds):.4f}") # mas especificos F1-Score para evaluar mejor el desempeño en clases desbalanceadas y poder solucionar problemas de precisión o exhaustividad que puedan aver con accuracy_score

print("\n--- Evaluación: Random Forest ---")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(f"F1-Score: {f1_score(y_test, rf_preds):.4f}")


# 5. Predicción para nuevos datos

# Creamos un registro sintético basado en las medias del dataset
new_data = pd.DataFrame([[55, 1, 1, 130, 250, 0, 1, 155, 0, 1.2, 1, 0, 2]], columns=X.columns) #parametros dados son ejemplos típicos de un paciente con riesgo moderado de enfermedad cardíaca (edad, tipo de dolor de pecho, presión arterial, colesterol, etc.) sacado de la media de cada variable en el dataset, ajustados para representar un caso realista.

dt_new_pred = dt_model.predict(new_data)
rf_new_pred = rf_model.predict(new_data)

print("\n--- Predicción para nuevo paciente ---")
print(f"Datos: {new_data.values.tolist()}")
print(f"Predicción Árbol de Decisión: {'Enfermo' if dt_new_pred[0] == 1 else 'Sano'}")
print(f"Predicción Random Forest: {'Enfermo' if rf_new_pred[0] == 1 else 'Sano'}")

# Discusión breve:
# Ambas predicciones tienen sentido si los valores ingresados (ej. age=55, cp=1, thalach=155) 
# son consistentes con los patrones de riesgo aprendidos por el modelo.
