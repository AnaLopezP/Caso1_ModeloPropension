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
X = data.drop('Mas_1_coche', axis=1)
y = data['Mas_1_coche']

# División del conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarización de las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Concatenar las características y la variable objetivo
X_combined = pd.DataFrame(X_train_scaled)
y_combined = pd.Series(y_train)
combined = pd.concat([X_combined, y_combined], axis=1)

# Separar las clases
majority = combined[combined[y_train.name] == 0]
minority = combined[combined[y_train.name] == 1]

# Submuestrear la clase mayoritaria
majority_downsampled = resample(
    majority,
    replace=False,
    n_samples=len(minority),
    random_state=42
)

# Combinar de nuevo
downsampled = pd.concat([majority_downsampled, minority])
X_train_balanced = downsampled.drop(y_train.name, axis=1)
y_train_balanced = downsampled[y_train.name]

# Comprobaciones de NaN e infinito
print("Valores NaN en X_train_balanced:", X_train_balanced.isnull().sum().sum())
print("Valores NaN en y_train_balanced:", y_train_balanced.isnull().sum())
print("Valores infinitos en X_train_balanced:", np.isinf(X_train_balanced).sum().sum())
print("Valores infinitos en y_train_balanced:", np.isinf(y_train_balanced).sum())

# Lidiar con NaN e infinito
X_train_balanced.fillna(0, inplace=True)  # O elige otro método adecuado
X_train_balanced = X_train_balanced[np.isfinite(X_train_balanced).all(axis=1)]
y_train_balanced = y_train_balanced[X_train_balanced.index]  # Alinear índices

# Imprimir la nueva distribución de clases
print(f"Nueva distribución de clases: {pd.Series(y_train_balanced).value_counts(normalize=True)}")

# Crear el modelo de Árbol de Decisión
model = tree.DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    ccp_alpha=0.01,
    random_state=42
)

# Entrenamiento del modelo con datos balanceados
model.fit(X_train_balanced, y_train_balanced)

# Realizar predicciones
y_pred = model.predict(X_test_scaled)

# Evaluación del modelo
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {precision * 100:.2f}%")

# Informe de clasificación para ver más detalles
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))

# Validación cruzada con los datos escalados
cross_val_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5)
print(f"\nPrecisión promedio con cross-validation sobre datos balanceados: {cross_val_scores.mean() * 100:.2f}%")

# Guardar el modelo entrenado
import joblib
joblib.dump(model, 'modelo_coches_decision_tree.pkl')


# Probar distintos valores de ccp_alpha para la poda
path = model.cost_complexity_pruning_path(X_train_scaled, y_train)
ccp_alphas = path.ccp_alphas

# Probar cada valor de ccp_alpha
print("\nPrueba de diferentes valores de ccp_alpha:")
for alpha in ccp_alphas:
    model_alpha = tree.DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(model_alpha, X_train_scaled, y_train, cv=5)
    print(f"Alpha: {alpha:.5f}, Precisión promedio: {scores.mean() * 100:.2f}%")