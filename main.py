# Imports y configuración inicial
import pandas as pd  # Librería para manipulación y análisis de datos
import numpy as np   # Librería para cálculos numéricos
import matplotlib.pyplot as plt  # Librería para visualización de gráficos
import seaborn as sns # Librería para visualización de gráficos
from sklearn.preprocessing import StandardScaler # Escalado de características
from sklearn.model_selection import train_test_split, cross_val_score # División de datos y validación cruzada
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score # Métricas de evaluación de modelos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # Modelos de aprendizaje automático
from sklearn.linear_model import LogisticRegression # Modelo de regresión logística
from sklearn.svm import SVC # Máquinas de vectores de soporte
from sklearn.decomposition import PCA # Análisis de componentes principales
from sklearn.pipeline import Pipeline # Construir flujos de trabajo de aprendizaje automático

# Cargar los datasets
df_OT = pd.read_csv('datasets/OnlyTrivial_dt.csv') # Cargar el dataset de casos triviales
df_ONT = pd.read_csv('datasets/OnlyNonTrivial_dt.csv') # Cargar el dataset de casos no triviales

# Seleccionar el dataset y crear una copia explícita
dataset = df_OT.copy() # Seleccionar el dataset de casos triviales y crear una copia para manipularlo

# Manejo de valores faltantes
missing_values = dataset.isnull().sum() # Contar los valores faltantes por columna
missing_values = missing_values[missing_values > 0] # Filtrar solo las columnas con valores faltantes
if not missing_values.empty:
    print("Columnas con valores faltantes y su cantidad respectiva:")
    print(missing_values)
    dataset = dataset.dropna() # Eliminar las filas con valores faltantes
else:
    print("No hay valores faltantes en el DataFrame.")

# Escalado de las características
columns_to_scale = ['cbo', 'cboModified', 'fanin', 'fanout', 'wmc', 'dit', 'noc', 'rfc', 'lcom', 'lcom*'] # Seleccionar las columnas a escalar
dataset[columns_to_scale] = dataset[columns_to_scale].astype('float64') # Asegurar que las columnas sean de tipo float64
scaler = StandardScaler() # Crear un objeto escaler
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale]) # Aplicar el escalado a las columnas seleccionadas

# Visualización de las métricas escaladas
dataset[columns_to_scale].hist(bins=30, figsize=(15, 10)) # Histogramas de las características escaladas
plt.show()
plt.figure(figsize=(10, 8))
sns.heatmap(dataset[columns_to_scale].corr(), annot=True, cmap='coolwarm') # Mapa de calor de correlaciones
plt.show()

# Reducción de dimensionalidad con PCA
pca = PCA(n_components=2) # Crear un objeto PCA con 2 componentes principales
pca_results = pca.fit_transform(dataset[columns_to_scale]) # Aplicar PCA a los datos escalados
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=dataset['refactoring'], cmap='viridis') # Graficar los datos en el espacio PCA
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# Separar características y variable objetivo
X = dataset[columns_to_scale] # Establecer las características (variables independientes)
Y = dataset['refactoring'] # Establecer la variable objetivo (variable dependiente)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # Dividir los datos en conjuntos de entrenamiento y prueba

# Definir y evaluar los modelos
models = { # Diccionario con los modelos a evaluar
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, solver="saga", random_state=42),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

def evaluate_model(model, X_train, X_test, y_train, y_test): # Función para entrenar y evaluar un modelo
    model.fit(X_train, y_train) # Entrenar el modelo con los datos de entrenamiento
    y_pred = model.predict(X_test) # Realizar predicciones con el modelo entrenado

    conf_matrix = confusion_matrix(y_test, y_pred) # Calcular la matriz de confusión
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues') # Visualizar la matriz de confusión
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusión - {model.__class__.__name__}')
    plt.show()

    print(f"Evaluación del modelo {model.__class__.__name__}:") # Imprimir el reporte de clasificación
    print(classification_report(y_test, y_pred))
    return { # Devolver las métricas de rendimiento del modelo
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred, average='weighted')
    }

results = {} # Diccionario para almacenar los resultados de los modelos
for model_name, model in models.items(): # Iterar a través de los modelos y evaluarlos
    print(f"Evaluando {model_name}")
    results[model_name] = evaluate_model(model, X_train, X_test, y_train, y_test)

results_df = pd.DataFrame(results).T # Convertir los resultados a un DataFrame para facilitar la comparación
print("Resultados de los modelos:")
print(results_df)

# Validación cruzada (opcional)
def cross_validate_model(model, x, y, cv=5): # Función para realizar validación cruzada
    scores = cross_val_score(model, x, y, cv=cv, scoring='accuracy') # Realizar la validación cruzada
    print(f"Accuracy promedio para {model.__class__.__name__}: {np.mean(scores):.3f}") # Imprimir la precisión promedio
    return scores

cross_validate_model(models["Random Forest"], X, Y) # Ejemplo de validación cruzada para el modelo Random Forest