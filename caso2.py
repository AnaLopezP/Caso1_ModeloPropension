#Pasamos al entrenamiento del modelo: Árbol de decisión

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#leemos el dataset
data = pd.read_csv('csvs/cars_numeros.csv', sep=';')

#Separamos las variables características de la objetivo
# variable obetivo Mas_1_coche
X = data.drop('Mas_1_coche', axis=1)
y = data['Mas_1_coche']

#Dividimos los datos en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Creamos el modelo
model = tree.DecisionTreeClassifier()

#Entrenamos el modelo
model.fit(X_train, y_train)

#Hacemos predicciones
y_pred = model.predict(X_test)

#Comprobamos la precisión 
print("Precisión: ", accuracy_score(y_test, y_pred))

#Guardamos el modelo
import joblib
joblib.dump(model, 'modelo_coches.pkl')



