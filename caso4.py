# Aquí vamos comparar con gráficas los resultados de predicciones_cars con Mas_1_coche de cars_numeros
from caso2 import X_train, model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Cargamos los datos
data = pd.read_csv('csvs/cars_numeros.csv', sep=';')
predicciones = pd.read_csv('csvs/predicciones_cars.csv', sep=';')

# Calculamos los porcentajes de cada clase en el dataset original
porcentaje_data = data['Mas_1_coche'].value_counts(normalize=True) * 100

# Calculamos los porcentajes de las predicciones
porcentaje_predicciones = predicciones['Predicciones'].value_counts(normalize=True) * 100

# Hacemos dos diagramas de barras para comparar los porcentajes
plt.figure(figsize=(10, 5))

# Diagrama de barras para 'Mas_1_coche' con porcentajes
plt.subplot(1, 2, 1)
ax1 = porcentaje_data.plot(kind='bar', title='Mas_1_coche (Porcentaje)')
plt.ylabel('Porcentaje (%)')

# Añadimos las etiquetas de porcentaje sobre cada barra
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.2f}%', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                 textcoords='offset points')

# Diagrama de barras para 'Predicciones' con porcentajes
plt.subplot(1, 2, 2)
ax2 = porcentaje_predicciones.plot(kind='bar', title='Predicciones (Porcentaje)')
plt.ylabel('Porcentaje (%)')

# Añadimos las etiquetas de porcentaje sobre cada barra
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.2f}%', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                 textcoords='offset points')

# Guardamos la gráfica en la carpeta 'img' con formato PNG
plt.savefig('img/comparacion_porcentajes.png')

# Si prefieres mostrar la gráfica en pantalla también
plt.show()

# Vemos que la proporcion de 1 y 0 es similar en ambos casos (el csv de entrenamiento y el utilizado para predecir), 
# osea que podemos deducir que el modelo ha funcionado bien
# varía en un 1%, lo cual es un margen aceptable

# Obtenemos la importancia de las características
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
plt.savefig('img/importancia_caracteristicas.png')
plt.show()




