# =============================================================================
# Tarea: Parte 2 - Implementación de modelos (Paso 2)
# Descripción: Análisis de importancia de variables y experimentos de impacto.
# Autor: Salvador
# Fecha: 19 de Marzo, 2026
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
# 1. Análisis de Importancia de Variables (Paso 1 models)
# -----------------------------------------------------------------------------
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)

# Obtener importancia
dt_importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("Importancia en Árbol de Decisión (Top 5):")
print(dt_importances.head())
print("\nImportancia en Random Forest (Top 5):")
print(rf_importances.head())

# Identificamos 3 variables relevantes: cp, ca, thal (basado en RF que suele ser más estable)
relevant_vars = ['cp', 'ca', 'thal']

# -----------------------------------------------------------------------------
# 2. Experimento: Limitar variables relevantes
# -----------------------------------------------------------------------------
# Caso A: Modelo original (todas las variables)
acc_original = accuracy_score(y_test, rf_model.predict(X_test))

# Caso B: Sin las 3 variables más importantes
X_limited = X.drop(relevant_vars, axis=1)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_limited, y, test_size=0.2, random_state=42)
rf_limited = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train_l, y_train_l)
acc_limited = accuracy_score(y_test_l, rf_limited.predict(X_test_l))

print(f"\nExactitud original: {acc_original:.4f}")
print(f"Exactitud eliminando {relevant_vars}: {acc_limited:.4f}")

# -----------------------------------------------------------------------------
# 3. Visualización del Árbol de Decisión
# -----------------------------------------------------------------------------
# Se visualiza el árbol para entender cómo toma decisiones
plt.figure(figsize=(20,10))
plot_tree(dt_model, feature_names=X.columns, class_names=['Sano', 'Enfermo'], filled=True, rounded=True, fontsize=10)
plt.title("Visualización del Árbol de Decisión - Paso 2")
plt.savefig(os.path.join(base_path, 'arbol_decision.png'))
print("\nVisualización guardada como 'arbol_decision.png'")

# -----------------------------------------------------------------------------
# 4. Análisis de resultados
# -----------------------------------------------------------------------------
# - Variables que influyen más: 'cp' (tipo de dolor de pecho), 'ca' (número de vasos), 'thal'.
# - Sentido: Médicamente estas variables son indicadores críticos de salud coronaria.
# - Complejidad: Al eliminar variables importantes, el modelo puede requerir más profundidad 
#   para compensar la pérdida de información, o simplemente perder capacidad predictiva.
