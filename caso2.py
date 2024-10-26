# ESTE ES EL QUE FUNCIONA BIEN

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import numpy as np

# Cargar los datos
data = pd.read_csv('csvs/cars_numeros.csv', sep=';')

# Separar las características y la variable objetivo
X = data.drop(['Mas_1_coche', 'Tiempo'], axis=1)
y = data['Mas_1_coche']

# División del conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de random forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)


# Entrenamiento del modelo 
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {precision * 100:.2f}%")

# Informe de clasificación para ver más detalles
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))

# Validación cruzada con los datos escalados
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"\nPrecisión promedio con cross-validation sobre datos balanceados: {cross_val_scores.mean() * 100:.2f}%")

# Guardar el modelo entrenado
import joblib
joblib.dump(model, 'modelo_coches_decision_tree.pkl')

# Hacemos la matriz de confusión
from sklearn.metrics import confusion_matrix

matriz_confu = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión:")
print(matriz_confu)

# Hacemos la curva ROC
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.show()

# Calcular el área bajo la curva ROC
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nÁrea bajo la curva ROC: {roc_auc:.2f}")

# Guardar la curva ROC
np.save('fpr.npy', fpr)
np.save('tpr.npy', tpr)
np.save('roc_auc.npy', roc_auc)

# Guardar la matriz de confusión
np.save('matriz_confu.npy', matriz_confu)

# Guardar la precisión
np.save('precision.npy', precision)

# Guardo las imágenes en la carpeta img
import os
os.makedirs('img', exist_ok=True)

# Guardamos la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.savefig('img/roc.png')

# Guardamos la matriz de confusión
import seaborn as sns



plt.figure()
sns.heatmap(matriz_confu, annot=True, cmap='Blues')
plt.xlabel('Predicciones')
plt.ylabel('Valores reales')
plt.title('Matriz de confusión')
plt.savefig('img/matriz_confusion.png')


