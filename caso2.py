'''#Pasamos al entrenamiento del modelo: Árbol de decisión

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#leemos el dataset
data = pd.read_csv('csvs/cars_numeros.csv', sep=';')

#Separamos las variables características de la objetivo
# variable obetivo Mas_1_coche
#X = data.drop('Mas_1_coche', axis=1)
X = data[['Tiempo', 'Revisiones', 'EDAD_COCHE', 'PRODUCTO', 'TIPO_CARROCERIA', 'Campanna3']]
y = data['Mas_1_coche']

#Dividimos los datos en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creamos el modelo
model = tree.DecisionTreeClassifier()

#Entrenamos el modelo
model.fit(X_train, y_train)

#Hacemos predicciones
y_pred = model.predict(X_test)

#Calculamos la precisión del modelo
precision = accuracy_score(y_test, y_pred)

# Mostramos la precisión
print('Precisión del modelo: ', precision*100, '%')

#Guardamos el modelo
import joblib
joblib.dump(model, 'modelo_coches.pkl')

# Ahora lo testeamos
# Cargamos el modelo
model = joblib.load('modelo_coches.pkl')

# Hacemos predicciones
y_pred = model.predict(X_test)

# Calculamos la precisión del modelo
precision = accuracy_score(y_test, y_pred)

# Mostramos la precisión
print('Precisión del modelo: ', precision*100, '%')
# Si la precisión es del 100% es que el modelo está sobreentrenaado, es decir, que se ha entrenado con los mismos datos con los que se ha testeado


'''

# Importamos las librerías necesarias
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Cargar los datos (Ajusta el path a tu archivo CSV)
data = pd.read_csv('csvs/cars_numeros.csv', sep=';')

# Separar las características (features) y la variable objetivo (target)
#X = data[['Tiempo', 'Revisiones', 'EDAD_COCHE', 'PRODUCTO', 'TIPO_CARROCERIA', 'Campanna3']]
X = data.drop('Mas_1_coche', axis=1)
y = data['Mas_1_coche']

# Miro cual es la distribucion de la variable 
# Revisa la distribución de la variable objetivo
print(y.value_counts())

# División del conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarización de las características (opcional, pero recomendado si hay variables con escalas diferentes)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo de Árbol de Decisión con limitación de profundidad para evitar sobreajuste
model = tree.DecisionTreeClassifier(max_depth=3, random_state=42)

# Entrenamiento del modelo
model.fit(X_train_scaled, y_train)

# Realizar predicciones
y_pred = model.predict(X_test_scaled)

# Evaluación del modelo
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {precision * 100:.2f}%")

# Informe de clasificación para ver más detalles
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))

# Validación cruzada para verificar que no hay sobreajuste
cross_val_scores = cross_val_score(model, X, y, cv=5)
print(f"\nPrecisión promedio con cross-validation: {cross_val_scores.mean() * 100:.2f}%")

# Guardar el modelo entrenado
import joblib
joblib.dump(model, 'modelo_coches_decision_tree.pkl')

