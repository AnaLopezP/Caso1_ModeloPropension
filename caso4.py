# Aquí vamos comparar con gráficas los resultados de predicciones_cars con Mas_1_coche de cars_numeros
#from caso2 import X_train, model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Cargamos los datos
data = pd.read_csv('csvs/cars_numeros.csv', sep=';')
predicciones = pd.read_csv('csvs/predicciones_cars.csv', sep=';')

#Hacemos dos diagramas de barras para comparar los resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
data['Mas_1_coche'].value_counts().plot(kind='bar', title='Mas_1_coche')
plt.subplot(1, 2, 2)
predicciones['Predicciones'].value_counts().plot(kind='bar', title='Predicciones')
plt.show()

# Vemos que la proporcion de 1 y 0 es similar en ambos casos (el csv de entrenamiento y el utilizado para predecir), 
# osea que podemos deducir que el modelo ha funcionado bien

'''# Obtenemos la importancia de las características
importances = model.feature_importances_

# Creamos un DataFrame para visualizar las características y su importancia
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Mostramos la importancia de las características
print(feature_importance_df)

# También podemos graficar la importancia de las características
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.title('Importancia de las características en el modelo Random Forest')
plt.xlabel('Importancia')
plt.gca().invert_yaxis()  # Invertimos el eje y para que la característica más importante esté en la parte superior
plt.show()
'''

# Calculamos la AUC-ROC para las predicciones
auc_score = roc_auc_score(data['Mas_1_coche'], predicciones['Predicciones'])
print(f'AUC-ROC: {auc_score:.4f}')

# También podemos generar la curva ROC
fpr, tpr, thresholds = roc_curve(data['Mas_1_coche'], predicciones['Predicciones'])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--')  # Curva aleatoria
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc='best')
plt.show()
