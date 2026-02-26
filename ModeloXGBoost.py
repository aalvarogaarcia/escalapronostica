from matplotlib.pyplot import clf
import numpy as np
import xgboost as xgb
import shap
import LecturaDatos
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import builtins

def set_model(datos_limpios, target_col: str, threshold: float = 0.05):
    # Aquí puedes configurar tu modelo XGBoost con los hiperparámetros deseados
    
    y = datos_limpios[target_col]
    X = datos_limpios.drop(columns=[target_col])

    X = pd.get_dummies(X, drop_first=True)  # Convertir variables categóricas a numéricas

    count_neg = (y==0).sum()
    count_pos = (y==1).sum()
    scale_pos_weight = count_neg / count_pos if count_pos > 0 else 1

    model_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 100,      # Número moderado de árboles
        'max_depth': 3,           # Árboles poco profundos (clave para N pequeño)
        'learning_rate': 0.05,    # Aprendizaje lento para mayor robustez
        'subsample': 0.8,         # Usar solo 80% de datos por árbol
        'colsample_bytree': 0.8,  # Usar solo 80% de variables por árbol
        'scale_pos_weight': scale_pos_weight, # Manejo de desbalance
        'use_label_encoder': False,
        'missing': np.nan         # Instrucción explícita de cómo leer NaNs
    }

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    return model



def training_skf(n_splits: int, shuffle: bool, random_state: int, model, X, y):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    auc_scores = []
    shap_values_list = []
    test_indices_list = [] 

    print("Iniciando Validación Cruzada...")

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Entrenar modelo
        clf = xgb.XGBClassifier(**model)
        clf.fit(X_train, y_train)
        
        # Predicción
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # Evaluar AUC en este pliegue
        auc = roc_auc_score(y_test, y_pred_proba)
        auc_scores.append(auc)
        
        # --- EXPLICABILIDAD (SHAP) ---
        # Guardamos la función float original
        original_float = builtins.float
        
        # Creamos un float parcheado que quita los corchetes si los detecta
        def patched_float(x):
            if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
                return original_float(x[1:-1]) # Extrae lo de adentro: '[0.5]' -> '0.5'
            return original_float(x)
            
        try:
            # Reemplazamos temporalmente float() en todo Python
            builtins.float = patched_float
            explainer = shap.TreeExplainer(clf)
        finally:
            # PASE LO QUE PASE, devolvemos a Python su float() original inmediatamente
            builtins.float = original_float
            
        shap_values = explainer.shap_values(X_test)
        
        shap_values_list.append(shap_values)
        test_indices_list.append(test_index)
    return auc_scores, shap_values_list, test_indices_list