# Tarea: Parte 4 - Análisis de predicciones individuales
# Descripción: Inspección de registros individuales y análisis de cambios.
# Gutierrez Hernandez Kevin Andrew
# Fecha: 19 de Marzo, 2026

#

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Configuración de rutas
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, 'heart_disease.csv')
df = pd.read_csv(csv_path)

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamos modelos (como en el Paso 1)
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)


# 1. Selección de tres registros (ejemplos del conjunto de prueba)

# Tomamos los índices 0, 10 y 20 del conjunto de prueba
indices = [0, 10, 20]
samples = X_test.iloc[indices]
actual_labels = y_test.iloc[indices]

print("--- Análisis de Registros Individuales ---")
for i, (idx, row) in enumerate(samples.iterrows()):
    print(f"\nRegistro {i+1} (Índice Original: {idx})")
    print(f"Valores de variables: {row.to_dict()}")
    
    dt_pred = dt_model.predict(samples.iloc[[i]])[0]
    rf_pred = rf_model.predict(samples.iloc[[i]])[0]
    
    print(f"Predicción Árbol de Decisión: {'Enfermo' if dt_pred == 1 else 'Sano'}")
    print(f"Predicción Random Forest: {'Enfermo' if rf_pred == 1 else 'Sano'}")
    print(f"Etiqueta Real: {'Enfermo' if actual_labels.iloc[i] == 1 else 'Sano'}")


# 2. Modificación de una variable importante ('cp' - dolor de pecho)

print("\n--- Experimento: Modificar variable 'cp' (Dolor de Pecho) ---")
sample_to_mod = samples.iloc[[0]].copy()
print(f"Original cp: {sample_to_mod['cp'].values[0]}")
# Cambiamos cp a un valor alto (ej. 3) si era bajo, o viceversa
sample_to_mod.at[sample_to_mod.index[0], 'cp'] = 3 if sample_to_mod['cp'].values[0] == 0 else 0

new_dt_pred = dt_model.predict(sample_to_mod)[0]
new_rf_pred = rf_model.predict(sample_to_mod)[0]

print(f"Nuevo cp: {sample_to_mod['cp'].values[0]}")
print(f"Nueva Predicción DT: {'Enfermo' if new_dt_pred == 1 else 'Sano'}")
print(f"Nueva Predicción RF: {'Enfermo' if new_rf_pred == 1 else 'Sano'}")

# Explicación breve:
# El cambio en 'cp' influye significativamente ya que es una de las variables con mayor 
# importancia detectada en el Paso 2. Al cambiar el tipo de dolor de pecho, el modelo 
# recorre una rama diferente del árbol de decisión, lo que puede alterar drásticamente 
# el resultado final.
