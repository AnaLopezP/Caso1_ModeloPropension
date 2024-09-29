# Primero hacemos una limpieza de los datos. 
# Importamos los datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importamos los datos
data = pd.read_csv('csvs/cars_input.csv', sep=';', encoding='UTF-8')
data.head()

#Eliminamos la columna CODE
data = data.drop(columns=['CODE'])

#Agregamos la columna EDAD_COCHE siendo años random, entre la columna QUEJA_CAC y COSTE_VENTA

# Generamos una columna de años aleatorios entre 'QUEJA_CAC' y 'COSTE_VENTA'
data['EDAD_COCHE'] = np.random.randint(1, 20, data.shape[0])

# Verificamos en qué posición está 'QUEJA_CAC'
posicion_queja = data.columns.get_loc('QUEJA_CAC')

# Insertamos la columna 'EDAD_COCHE' justo después de 'QUEJA_CAC' y antes de 'COSTE_VENTA'
data.insert(posicion_queja + 1, 'EDAD_COCHE', data.pop('EDAD_COCHE'))





#-------------------------LIMPIEZA DE LOS DATOS--------------------------------
# Buscamos repetidos
# Buscamos nulos (null y vacios)
# Buscamos valores atipicos
# Buscamos datos inconsistentes

# Buscamos repetidos
print("Numero de filas duplicadas: ")
print(data.duplicated().sum())
# quitamos los duplicados
data = data.drop_duplicates()


# Buscamos datos inconsistentes
print("Datos inconsistentes: ")
print(data['GENERO'].unique())
print(data['ESTADO_CIVIL'].unique())
print(data['Zona_Renta'].unique())
print(data['Averia_grave'].unique())

# Buscamos nulos (null y vacios)
print("Numero de nulos: ")
print(data.isnull().sum())
# vemos que hay 890 en estado civil, 860 en genero, 13178 en zona_renta y hay 1 en averia
# Con el estado civil, vamos a sustituir los nulos por "No informado"
# Con el genero, vamos a sustituir los nulos por "Otro"
# Con la zona_renta, vamos a sustituir los nulos por "No informado"
# Con la avería, como solo es un nulo, podemos eliminar la fila

data['ESTADO_CIVIL'] = data['ESTADO_CIVIL'].fillna('OTROS')
data['GENERO'] = data['GENERO'].fillna('Otro')
data['Zona_Renta'] = data['Zona_Renta'].fillna('Otros')
data = data.dropna(subset=['Averia_grave'])

# Comprobamos que ha funcionado:
print("Numero de nulos: ")
print(data.isnull().sum())

# Buscamos valores atipicos
print("Valores atipicos: ")
print(data.describe())
# Aunque igual hay algun dato más alto que otro, no consideramos que sea atípico, ya que se repite varias veces

# Guardamos el archivo limpio
data.to_csv('csvs/cars_input_clean.csv', sep=';', index=False)
print("Archivo guardado")

# Separamos en datos categóricos y numéricos
data_cat = data.select_dtypes(include=['object'])
data_num = data.select_dtypes(exclude=['object'])

print(data_cat.head())
print(data_num.head())

#-------------------------TRANSFORMACIÓN DE LOS DATOS--------------------------------

#Cambiamos las variables categoricas a numericas
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_cat = data_cat.apply(le.fit_transform)
print(data_cat.head())

#valores únicos de las variables categóricas
for col in data_cat.columns:
    print(col)
    print(data_cat[col].unique())

#guardamos los datos
data_cat.to_csv('csvs/data_cars_input.csv', sep=';', index=False)

#Juntamos los datos
data = pd.concat([data_cat, data_num], axis=1)
data.to_csv('csvs/cars_input_numeros.csv', sep=';', index=False)

#-------------------------REPRESENTACIÓN DE LOS DATOS--------------------------------

#Representamos los datos, hacemos histogramas, diagramas de dispersión, mapa de correlación
#Histograma
data.hist()
plt.show()

#Mapa de correlación de los datos
correlacion = data.corr()
sns.heatmap(correlacion, annot=True)
plt.show()

# Creo una carpeta para guardar las imágenes
import os
os.makedirs('img', exist_ok=True)

# Guardamos el histograma
plt.figure(figsize=(10, 5))
data.hist()
plt.tight_layout()
plt.savefig('img/histograma.png')

# Guardamos el mapa de correlación
plt.figure(figsize=(10, 5))
sns.heatmap(correlacion, annot=True)
plt.tight_layout()
plt.savefig('img/correlacion.png')




