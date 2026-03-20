# =============================================================================
# Tarea: Parte 2 - Implementación de modelos (Paso 3)
# Descripción: Optimización de hiperparámetros con GridSearchCV.
# Autor: Salvador
# Fecha: 19 de Marzo, 2026
# =============================================================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configuración de rutas
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, 'heart_disease.csv')
df = pd.read_csv(csv_path)

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------------------------------------
# 1. Definición del modelo y búsqueda de hiperparámetros
# -----------------------------------------------------------------------------
rf = RandomForestClassifier(random_state=42)

# Parámetros a probar:
# - n_estimators: número de árboles.
# - max_depth: profundidad máxima de los árboles.
# - min_samples_split: mínimo de muestras para dividir un nodo.
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

print("Iniciando GridSearchCV para optimizar Random Forest...")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 2. Resultados de la optimización
# -----------------------------------------------------------------------------
print("\n--- Resultados de GridSearchCV ---")
print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
print(f"Mejor score (cross-validation): {grid_search.best_score_:.4f}")

# Comparación antes y después
# Modelo original (de Paso 1)
rf_original = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)
acc_original = accuracy_score(y_test, rf_original.predict(X_test))

# Modelo optimizado
rf_best = grid_search.best_estimator_
acc_optimized = accuracy_score(y_test, rf_best.predict(X_test))

print("\n--- Comparación de Desempeño ---")
print(f"Exactitud antes de optimización: {acc_original:.4f}")
print(f"Exactitud después de optimización: {acc_optimized:.4f}")

# -----------------------------------------------------------------------------
# 3. Discusión breve
# -----------------------------------------------------------------------------
# La optimización permite encontrar el balance ideal entre sesgo (bias) y varianza (variance),
# mejorando la capacidad de generalización del modelo frente a datos no vistos.
