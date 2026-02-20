import matplotlib.pyplot as plt
import numpy as np
import ModeloXGBoost

def plot_shap_summary(shap_values_list, X, test_indices_list):
    # Reconstruir el dataset completo ordenado como se validó
    X_test_all = X.iloc[test_indices_list]
    shap_values_all = np.concatenate(shap_values_list)


    # 1. Gráfico de resumen (Beeswarm plot)
    # Muestra las variables más importantes y cómo sus valores altos/bajos impactan
    plt.figure()
    ModeloXGBoost.shap.summary_plot(shap_values_all, X_test_all, show=False)
    plt.title("Impacto de las variables en la Mortalidad Intrahospitalaria")
    plt.tight_layout()
    plt.show()

    # 2. Gráfico de Barras (Importancia absoluta)
    plt.figure()
    ModeloXGBoost.shap.summary_plot(shap_values_all, X_test_all, plot_type="bar", show=False)
    plt.title("Variables con mayor peso predictivo")
    plt.tight_layout()
    plt.show()