# =============================================================================
# Tarea: Parte 3 - Manejo de datos faltantes
# Descripción: Comparación de modelos con y sin imputación en dataset messy.
# Autor: Salvador
# Fecha: 19 de Marzo, 2026
# Fuente: Horse Colic Dataset (UCI)
# =============================================================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Configuración de rutas
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, 'horse_colic.csv')

# Cargamos el dataset indicando que '?' representa valores nulos (NaN)
df = pd.read_csv(csv_path, na_values='?')

# Preprocesamiento: 'outcome' es la etiqueta (columna 23, índice 22)
# Eliminamos filas donde la etiqueta sea nula
df = df.dropna(subset=['outcome'])

X = df.drop('outcome', axis=1)
y = df['outcome']

# -----------------------------------------------------------------------------
# a) Entrenamiento usando solo variables SIN valores faltantes
# -----------------------------------------------------------------------------
cols_no_na = X.columns[X.notna().all()].tolist()
print(f"Columnas sin valores faltantes: {cols_no_na}")

X_no_na = X[cols_no_na]
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_no_na, y, test_size=0.2, random_state=42)

dt_a = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train_a, y_train_a)
rf_a = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train_a, y_train_a)

acc_dt_a = accuracy_score(y_test_a, dt_a.predict(X_test_a))
acc_rf_a = accuracy_score(y_test_a, rf_a.predict(X_test_a))

# -----------------------------------------------------------------------------
# b) Entrenamiento con IMPUTACIÓN (SimpleImputer)
# -----------------------------------------------------------------------------
# Imputamos con la media para variables numéricas
imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

dt_b = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train_b, y_train_b)
rf_b = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train_b, y_train_b)

acc_dt_b = accuracy_score(y_test_b, dt_b.predict(X_test_b))
acc_rf_b = accuracy_score(y_test_b, rf_b.predict(X_test_b))

# -----------------------------------------------------------------------------
# 4. Comparación de Resultados
# -----------------------------------------------------------------------------
print("\n--- Comparativa de Resultados ---")
print(f"Sin imputación (solo variables completas):")
print(f"  - Decision Tree: {acc_dt_a:.4f}")
print(f"  - Random Forest: {acc_rf_a:.4f}")
print(f"Con imputación (todas las variables):")
print(f"  - Decision Tree: {acc_dt_b:.4f}")
print(f"  - Random Forest: {acc_rf_b:.4f}")

# Predicción para nuevos datos
new_horse = X_imputed.iloc[[0]].copy() # Usamos el primer registro como base
print("\n--- Predicción para un nuevo caso (Horse Colic) ---")
pred_a = rf_a.predict(new_horse[cols_no_na])
pred_b = rf_b.predict(new_horse)
print(f"Predicción (Sin imputación): {pred_a[0]}")
print(f"Predicción (Con imputación): {pred_b[0]}")
