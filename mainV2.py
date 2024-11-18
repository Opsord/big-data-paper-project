# Imports y configuración inicial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Configuración global de seeds para reproducibilidad
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configuración básica de visualización
sns.set_theme()  # Esto configura el estilo de seaborn de manera segura


def load_and_prepare_data(filepath, random_state=RANDOM_STATE):
    """
    Carga y prepara los datos iniciales
    """
    try:
        dataset = pd.read_csv(filepath)
        print(f"Dimensiones del dataset: {dataset.shape}")
        return dataset
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filepath}")
        raise


def handle_missing_values(dataset):
    """
    Maneja valores faltantes y retorna estadísticas
    """
    missing_values = dataset.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print("Columnas con valores faltantes:")
        print(missing_values)
        return dataset.dropna()
    print("No hay valores faltantes en el DataFrame.")
    return dataset


def scale_features(dataset, columns_to_scale):
    """
    Escala características numéricas usando StandardScaler
    """
    scaler = StandardScaler()
    dataset_scaled = dataset.copy()
    dataset_scaled[columns_to_scale] = dataset_scaled[columns_to_scale].astype('float64')
    dataset_scaled[columns_to_scale] = scaler.fit_transform(dataset_scaled[columns_to_scale])
    return dataset_scaled, scaler


def visualize_distributions(data, columns):
    """
    Visualiza distribuciones de características
    """
    n_cols = 2
    n_rows = (len(columns) + 1) // 2
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 4 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, column in enumerate(columns):
        row = idx // n_cols
        col = idx % n_cols
        sns.histplot(data=data, x=column, ax=axes[row, col], kde=True)
        axes[row, col].set_title(f'Distribución de {column}')

    # Ocultar subplots vacíos si los hay
    for idx in range(len(columns), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()


def perform_pca(data, n_components=2, random_state=RANDOM_STATE):
    """
    Realiza PCA y visualiza resultados
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_results = pca.fit_transform(data)

    explained_variance = pca.explained_variance_ratio_
    print(f"Varianza explicada por componentes: {explained_variance}")

    # Visualizar resultados PCA
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.5)
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} varianza explicada)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} varianza explicada)')
    plt.title('Visualización PCA')
    plt.show()

    return pca_results, pca


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Evalúa un modelo individual y retorna métricas
    """
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Crear matriz de confusión
        plt.figure(figsize=(8, 6))
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.show()

        print(f"\nReporte de clasificación para {model_name}:")
        print(classification_report(y_test, y_pred))

        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred, average='weighted')
        }
    except Exception as e:
        print(f"Error al evaluar el modelo {model_name}: {str(e)}")
        return None


class ModelEvaluator:
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.models = {
            "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Random Forest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
            "XGBoost": XGBClassifier(random_state=random_state, n_jobs=-1)
        }

    def cross_validate_models(self, X, y, cv=5):
        """
        Realiza validación cruzada estratificada para todos los modelos
        """
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
                cv_results[name] = {
                    "mean_accuracy": np.mean(scores),
                    "std_accuracy": np.std(scores)
                }
            except Exception as e:
                print(f"Error en validación cruzada para {name}: {str(e)}")
                cv_results[name] = {"error": str(e)}

        return pd.DataFrame(cv_results).T


def main():
    try:
        # Definir columnas a escalar
        columns_to_scale = ['cbo', 'cboModified', 'fanin', 'fanout', 'wmc', 'dit', 'noc', 'rfc', 'lcom', 'lcom*']

        # Cargar datasets
        print("Cargando datasets...")
        df_ot = load_and_prepare_data('datasets/OnlyTrivial_dt.csv')
        df_ont = load_and_prepare_data('datasets/OnlyNonTrivial_dt.csv')

        # Seleccionar dataset para el análisis
        dataset = df_ot.copy()

        # Manejar valores faltantes
        print("\nVerificando valores faltantes...")
        dataset = handle_missing_values(dataset)

        # Escalar características
        print("\nEscalando características...")
        dataset_scaled, _ = scale_features(dataset, columns_to_scale)

        # Visualizar distribuciones
        print("\nGenerando visualizaciones de distribuciones...")
        visualize_distributions(dataset_scaled, columns_to_scale)

        # Preparar datos para modelado
        X = dataset_scaled[columns_to_scale]
        y = dataset['refactoring']

        # Realizar PCA
        print("\nRealizando análisis PCA...")
        pca_results, _ = perform_pca(X)

        # Dividir datos
        print("\nDividiendo datos en train y test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        # Evaluar modelos
        print("\nEvaluando modelos...")
        evaluator = ModelEvaluator()
        results = {}

        for name, model in evaluator.models.items():
            print(f"\nEvaluando {name}...")
            results[name] = evaluate_model(model, X_train, X_test, y_train, y_test, name)

        # Mostrar resultados de validación cruzada
        print("\nRealizando validación cruzada...")
        cv_results = evaluator.cross_validate_models(X, y)
        print("\nResultados de validación cruzada:")
        print(cv_results)

    except Exception as e:
        print(f"Error en la ejecución principal: {str(e)}")


if __name__ == "__main__":
    main()