import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap

def plot_shap_summary(shap_values_list, X, test_indices_list):
    # --- APLICAR ESTILO SEABORN ---
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # --- PREPARACIÓN DE DATOS ---
    test_indices_all = np.concatenate(test_indices_list)
    X_test_all = X.iloc[test_indices_all]
    shap_values_all = np.concatenate(shap_values_list)

    # ----------------------------------------
    # Gráfico 1: Resumen (Beeswarm plot)
    # ----------------------------------------
    plt.figure(figsize=(10, 6))
    
    # SHAP dibuja primero
    shap.summary_plot(shap_values_all, X_test_all, show=False, max_display=12)
    
    # CAPTURAMOS EL EJE ACTUAL Y LO MOVEMOS ARRIBA
    ax1 = plt.gca()
    ax1.xaxis.tick_top()                    # Mueve los números (ticks) arriba
    ax1.xaxis.set_label_position('top')     # Mueve el texto del eje X arriba
    
    # Aumentamos el 'pad' a 40 para que el título no se superponga con el eje X que acabamos de subir
    plt.title("Impacto de las variables en el Pronóstico (SHAP)", pad=40, fontweight='bold')
    plt.tight_layout()
    
    fig1 = plt.gcf() 

    # ----------------------------------------
    # Gráfico 2: Barras (Importancia absoluta)
    # ----------------------------------------
    plt.figure(figsize=(10, 6))
    
    # SHAP dibuja primero
    shap.summary_plot(shap_values_all, X_test_all, plot_type="bar", show=False, max_display=12)
    
    # CAPTURAMOS EL EJE ACTUAL Y LO MOVEMOS ARRIBA
    ax2 = plt.gca()
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top')
    
    plt.xlabel("Impacto Promedio Absoluto en la Predicción", fontsize=12)
    plt.title("Top 12 Variables con Mayor Peso Predictivo", pad=40, fontweight='bold')
    
    plt.tight_layout()
    fig2 = plt.gcf()
    
    # Restaurar estilo
    sns.reset_orig()

    return fig1, fig2