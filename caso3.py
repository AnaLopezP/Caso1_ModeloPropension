
#cargamos los datos 
import pandas as pd
data = pd.read_csv('csvs/cars_input.csv', sep=';', encoding='UTF-8')


# Separamos en datos categóricos y numéricos
data_cat = data.select_dtypes(include=['object'])
data_num = data.select_dtypes(exclude=['object'])

#Transformamos las variables categóricas a numéricas
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_cat = data_cat.apply(le.fit_transform)
print(data_cat.head())




#Cargamos el modelo
import joblib
modelo = joblib.load('modelo_coches_decision_tree.pkl')

#Predicciones
predicciones = modelo.predict(data)

#Guardamos los resultados
resultados = pd.DataFrame(predicciones, columns=['Predicciones'])

#Añadimos las predicciones al csv
data['Predicciones'] = predicciones

#Guardamos los resultados en un csv
data.to_csv('csvs/predicciones_cars.csv', sep=';', index = False)



