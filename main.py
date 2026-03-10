from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import pandas as pd
import os
import io
import numpy as np
from scipy import stats

# Importar tus módulos
import LecturaDatos
import ModeloXGBoost
import PresentacionClinica
from sklearn.preprocessing import LabelEncoder

# Configuración de la página
st.set_page_config(page_title="Escala Pronóstica Dinámica", layout="wide")

st.title("⚕️ Escala Pronóstica y Análisis de Datos Clínicos")

# ==========================================
# 0. INICIALIZACIÓN DE VARIABLES EN SESIÓN
# ==========================================
# Esto evita que los datos se borren al interactuar con la app
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None
if 'df_clean' not in st.session_state:
    st.session_state['df_clean'] = None

# Crear las 3 pestañas
tab1, tab2, tab3, tab4 = st.tabs(["📁 1. Carga", "⚙️ 2. Preprocesamiento", "📊 3. Análisis Bivariado", "🚀 4. Modelado Predictivo"])

## ==========================================
# PESTAÑA 1: CARGA DE DATOS
# ==========================================

with tab1:
    st.header("Carga de Archivo de Datos")
    uploaded_file = st.file_uploader("Sube tu archivo de datos (.sav o .csv)", type=["sav", "csv"])
    


    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get('curret_file_id') != file_id:
            with st.spinner("Leyendo datos..."):
                file_ext = uploaded_file.name.split('.')[-1].lower()

                # Cargar los datos según la extensión
                if file_ext == 'sav':
                    temp_path = "temp_datos.sav"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    df_temp = LecturaDatos.leer_datos(temp_path)
                    os.remove(temp_path)

                elif file_ext == 'csv':
                    df_temp = pd.read_csv(uploaded_file)

                # === LA SOLUCIÓN AQUÍ ===
                # Convertir columnas tipo 'category' a 'object' (texto) para evitar el error de PyArrow
                if df_temp is not None:
                    for col in df_temp.select_dtypes(['category']).columns:
                        df_temp[col] = df_temp[col].astype(str)

                    # Guardar en session_state
                    st.session_state['df_raw'] = df_temp

                    st.session_state['df_clean'] = df_temp.copy()  # Inicialmente limpio es igual a crudo, se limpiará en la pestaña 2
                    st.session_state['curret_file_id'] = file_id

        # Mostrar resultados
        if st.session_state['df_raw'] is not None:
            st.success(f"Archivo cargado exitosamente. ({st.session_state['df_raw'].shape[0]} filas, {st.session_state['df_raw'].shape[1]} columnas)")
            st.dataframe(st.session_state['df_raw'].head().astype(str))
            
# ==========================================
# PESTAÑA 2: PREPROCESAMIENTO DINÁMICO
# ==========================================
with tab2:
    st.header("Exploración y Limpieza de Datos")
    
    if st.session_state['df_raw'] is not None:
        df = st.session_state['df_raw']
        columnas_disponibles = df.columns.tolist()
        
        col_info1, col_info2 = st.columns([2, 1])
        with col_info1:
            st.subheader("Información de las columnas (Valores nulos y tipos)")
            
            resumen_nulos = pd.DataFrame({
                'Tipo': df.dtypes.astype(str), 
                'Nulos': df.isnull().sum(),
                '% Nulos': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(resumen_nulos)

        with col_info2:
            st.subheader("Configuración de Limpieza")
            st.markdown("Selecciona los parámetros para depurar tu base de datos:")
            
            cols_missing_info = st.multiselect(
                "Columnas para 'Missing as Information'", 
                options=columnas_disponibles,
                help="Los valores nulos en estas columnas se tratarán como una categoría/información válida."
            )
            
            cols_to_drop = st.multiselect(
                "Columnas a ELIMINAR (cols_to_drop)", 
                options=columnas_disponibles
            )
            
            cols_to_save = st.multiselect(
                "Columnas a GUARDAR (ignorar threshold) (cols_to_save)", 
                options=columnas_disponibles,
                help="Ejemplo: Tu variable objetivo 'MORTALIDAD'. No se eliminará aunque tenga muchos nulos."
            )
            
            if set(cols_to_drop).intersection(set(cols_to_save)):
                st.warning("⚠️ Has seleccionado la misma columna para eliminar y para guardar. El sistema le dará prioridad a GUARDAR.")


            threshold_missing = st.slider("Umbral de Nulos (Threshold)", 0.0, 1.0, 0.05, 0.01)
            
            if st.button("Aplicar Limpieza", use_container_width=True):
                with st.spinner("Limpiando datos..."):
                    df_temp = df.copy()
                    
                    # 1. Aplicar missing as information (adaptado si tu función acepta lista, si no, lo hacemos iterativo)
                    for col in cols_missing_info:
                        df_temp = LecturaDatos.missing_as_information(df_temp, col)
                    
                    # 2. Aplicar limpieza general
                    df_limpio = LecturaDatos.limpieza_datos(
                        df_temp, 
                        threshold=threshold_missing,
                        cols_to_drop_manual=cols_to_drop,
                        cols_to_save=cols_to_save
                    )
                    
                    st.session_state['df_clean'] = df_limpio
                    st.success("¡Datos limpios correctamente!")

        # Mostrar resultados y botón de descarga si ya se limpió
        if st.session_state['df_clean'] is not None:
            st.markdown("---")
            st.subheader("Vista previa de Datos Limpios")
            st.write(f"Dimensiones finales: {st.session_state['df_clean'].shape[0]} filas, {st.session_state['df_clean'].shape[1]} columnas")
            st.dataframe(st.session_state['df_clean'].head())
            
            # Botón para descargar el CSV limpio
            csv_data = st.session_state['df_clean'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Base de Datos Limpia (CSV)",
                data=csv_data,
                file_name='datos_limpios.csv',
                mime='text/csv'
            )

    else:
        st.info("Por favor, carga un archivo en la Pestaña 1 primero.")


# ==========================================
# PESTAÑA 3: ANÁLISIS BIVARIADO (TABLA 1 CLÍNICA)
# ==========================================
with tab3:
    st.header("📊 Análisis Bivariado (Características Basales)")
    
    if st.session_state.get('df_clean') is not None:
        # Usamos .copy() para que estas transformaciones visuales no afecten al modelo XGBoost
        df_desc = st.session_state['df_clean'].copy()
        
        with st.spinner("Optimizando tipos de datos y agrupando categorías..."):
            # --- 1. RESCATE DE VARIABLES NUMÉRICAS ---
            for col in df_desc.columns:
                if df_desc[col].dtype == 'object':
                    # Intenta convertir a número. Los textos puros se volverán NaN
                    temp_col = pd.to_numeric(df_desc[col], errors='coerce')
                    
                    # Si conseguimos convertir la gran mayoría (ej. >80% de los datos que no eran nulos originalmente)
                    # Asumimos que es numérica y la reemplazamos.
                    nulos_originales = df_desc[col].isna().sum()
                    nulos_nuevos = temp_col.isna().sum()
                    
                    if (len(df_desc) - nulos_nuevos) >= (len(df_desc) - nulos_originales) * 0.8:
                        if (len(df_desc) - nulos_nuevos) > 0: # Para evitar columnas vacías
                            df_desc[col] = temp_col

            # --- 2. AGRUPACIÓN DE CATEGORÍAS MINORITARIAS ---
            # Solo aplica a variables que quedaron como object o category
            cols_categoricas = df_desc.select_dtypes(include=['object', 'category']).columns
            for col in cols_categoricas:
                if df_desc[col].nunique() > 3:
                    frecuencias = df_desc[col].value_counts(normalize=True)
                    
                    # Si existe al menos un grupo dominante (>30%)
                    if frecuencias.max() >= 0.30:
                        # Identificamos grupos que representan menos del 5% (puedes ajustar este 0.05)
                        grupos_minoritarios = frecuencias[frecuencias < 0.05].index
                        
                        if len(grupos_minoritarios) > 0:
                            # Agrupamos todos los minoritarios bajo la etiqueta 'Otros'
                            df_desc[col] = df_desc[col].replace(grupos_minoritarios, 'Otros')

        st.markdown("""
        Selecciona una variable objetivo para dividir a tus pacientes en grupos.
        * **Numéricas**: Se muestran como *Mediana (Rango Intercuartílico)* (Tests: Mann-Whitney / Kruskal-Wallis).
        * **Categóricas**: Se muestran como *Frecuencia (%)* (Test: Chi-cuadrado).
        * *Nota: Las categorías muy pequeñas (<5%) se han agrupado en "Otros" automáticamente para dar validez estadística.*
        """)
        
        cols_potenciales = [c for c in df_desc.columns if df_desc[c].nunique() < 10 or df_desc[c].dtype == 'object']
        target_bivariado = st.selectbox("🎯 Variable de Agrupación", options=cols_potenciales)
        
        if st.button("Generar Tabla Descriptiva", use_container_width=True):
            with st.spinner("Calculando estadísticas y p-valores no paramétricos..."):
                resultados = []
                grupos = sorted([g for g in df_desc[target_bivariado].unique() if pd.notna(g)])
                
                for col in df_desc.columns:
                    if col == target_bivariado: continue
                    
                    is_num = pd.api.types.is_numeric_dtype(df_desc[col])
                    if is_num and set(df_desc[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                        is_num = False
                        
                    if is_num and df_desc[col].nunique() > 5:
                        row_dict = {'Variable': f"**{col}** (mediana [RIC])"}
                        grupos_data = [df_desc[df_desc[target_bivariado] == g][col].dropna() for g in grupos]
                        
                        tot = df_desc[col].dropna()
                        row_dict['Total'] = f"{tot.median():.1f} ({tot.quantile(0.25):.1f}-{tot.quantile(0.75):.1f})" if not tot.empty else "-"
                        
                        for g, data in zip(grupos, grupos_data):
                            if len(data) > 0:
                                row_dict[f'Grupo {g}'] = f"{data.median():.1f} ({data.quantile(0.25):.1f}-{data.quantile(0.75):.1f})"
                            else:
                                row_dict[f'Grupo {g}'] = "-"
                                
                        try:
                            valid_data = [d for d in grupos_data if len(d) > 0]
                            if len(valid_data) == 2:
                                _, p = stats.mannwhitneyu(valid_data[0], valid_data[1], alternative='two-sided')
                            elif len(valid_data) > 2:
                                _, p = stats.kruskal(*valid_data)
                            else:
                                p = np.nan
                            row_dict['Valor p'] = f"{p:.3f}" if p >= 0.001 else "<0.001"
                        except:
                            row_dict['Valor p'] = "-"
                            
                        resultados.append(row_dict)
                        
                    else:
                        row_dict = {'Variable': f"**{col}** n (%)", 'Total': "", 'Valor p': ""}
                        for g in grupos: row_dict[f'Grupo {g}'] = ""
                        
                        try:
                            contingency = pd.crosstab(df_desc[col], df_desc[target_bivariado])
                            if contingency.empty or contingency.shape[0] < 2:
                                p = np.nan
                            else:
                                chi2, p, dof, ex = stats.chi2_contingency(contingency)
                            row_dict['Valor p'] = f"{p:.3f}" if p >= 0.001 else "<0.001"
                        except:
                            row_dict['Valor p'] = "-"
                            
                        resultados.append(row_dict)
                        
                        # Alfabético, pero poniendo "Otros" al final si existe
                        valores_unicos =[c for c in df_desc[col].unique() if pd.notna(c)]
                        try:
                            categorias = sorted(valores_unicos)
                        except:
                            st.warning(f"⚠️ La columna '{col}' contiene tipos de datos mezclados (texto y números). Se han convertido a texto para poder mostrarlos.")
                            categorias = sorted([str(c) for c in valores_unicos])
                            

                        if 'Otros' in categorias:
                            categorias.remove('Otros')
                            categorias.append('Otros')

                        for cat in categorias:
                            cat_row = {'Variable': f"  - {cat}", 'Valor p': ""}
                            
                            n_tot = (df_desc[col] == cat).sum()
                            pct_tot = (n_tot / len(df_desc[col].dropna()) * 100) if len(df_desc[col].dropna())>0 else 0
                            cat_row['Total'] = f"{n_tot} ({pct_tot:.1f}%)"
                            
                            for g in grupos:
                                mask_g = (df_desc[target_bivariado] == g)
                                n_g = ((df_desc[col] == cat) & mask_g).sum()
                                total_g = mask_g.sum()
                                pct_g = (n_g / total_g * 100) if total_g > 0 else 0
                                cat_row[f'Grupo {g}'] = f"{n_g} ({pct_g:.1f}%)"
                                
                            resultados.append(cat_row)
                            
                df_resultado = pd.DataFrame(resultados)
                st.dataframe(df_resultado, use_container_width=True)
                
                csv_bivariado = df_resultado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Tabla Descriptiva (CSV)",
                    data=csv_bivariado,
                    file_name='tabla_bivariada.csv',
                    mime='text/csv'
                )
    else:
        st.warning("⚠️ No hay datos limpios. Por favor, completa la limpieza en la **Pestaña 2**.")


# ==========================================
# PESTAÑA 4: MODELADO PREDICTIVO Y ESCALA PRONÓSTICA
# ==========================================
with tab4:
    st.header("🚀 Modelado Predictivo y Escala Pronóstica")
    
    if st.session_state.get('df_clean') is not None:
        df_model = st.session_state['df_clean']
        
        # --- 1. SELECCIÓN DE LA VARIABLE OBJETIVO ---
        st.subheader("1. Configuración de la Predicción")
        columnas_modelo = df_model.columns.tolist()
        
        # Intentar preseleccionar 'MORTALIDAD' si existe, si no, elige la primera
        idx_defecto = columnas_modelo.index('MORTALIDAD') if 'MORTALIDAD' in columnas_modelo else 0
        target_col = st.selectbox("🎯 Selecciona la Variable Objetivo (Target)", options=columnas_modelo, index=idx_defecto)
        
        # --- LÓGICA DEL HISTORIAL ---
        # Inicializamos el historial y el registro del último target analizado
        if 'historial_modelos' not in st.session_state:
            st.session_state['historial_modelos'] = []
        if 'last_target' not in st.session_state:
            st.session_state['last_target'] = target_col
            
        # Si el usuario cambia la variable objetivo, limpiamos el historial comparativo
        if st.session_state['last_target'] != target_col:
            st.session_state['historial_modelos'] = []
            st.session_state['last_target'] = target_col
            st.info(f"Historial reiniciado. Evaluando nueva variable objetivo: {target_col}")

        # --- 2. PANEL DE HIPERPARÁMETROS ---
        st.markdown("---")
        st.subheader("2. Ajuste de Hiperparámetros (XGBoost)")
        
        col_param1, col_param2, col_param3 = st.columns(3)
        with col_param1:
            n_estimators = st.slider("Número de Árboles (n_estimators)", 10, 500, 100, step=10)
            max_depth = st.slider("Profundidad Máxima (max_depth)", 1, 10, 3)
        with col_param2:
            learning_rate = st.selectbox("Tasa de Aprendizaje (learning_rate)", [0.01, 0.05, 0.1, 0.2, 0.3], index=1)
            subsample = st.slider("Muestras por árbol (subsample)", 0.1, 1.0, 0.8, step=0.1)
        with col_param3:
            n_splits = st.number_input("Particiones (K-Folds)", min_value=2, max_value=10, value=5)

        # --- 3. ENTRENAMIENTO ---
        if st.button("🚀 Entrenar Modelo y Generar Escala", use_container_width=True):
            with st.spinner("Entrenando modelo con validación cruzada y extrayendo interpretabilidad (SHAP)..."):
                
                # --- NUEVO: Preparar y codificar 'y' ---
                y_raw = df_model[target_col]
                
                # Si la variable objetivo es texto o categórica, la pasamos a 0 y 1
                if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
                    le = LabelEncoder()
                    y = pd.Series(le.fit_transform(y_raw), index=y_raw.index)
                    # Opcional: Mostrar qué significa cada número
                    st.caption(f"*(ℹ️ Internamente se codificó: **{le.classes_[0]} = 0** y **{le.classes_[1]} = 1**)*")
                else:
                    y = y_raw
                
                # Preparar X
                X = df_model.drop(columns=[target_col])
                X = pd.get_dummies(X, drop_first=True)
                
                scaler = StandardScaler()
                cols_num = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns

                #Filtrar para no escalar variables binarias (0/1)
                cols_to_scale = [c for c in cols_num if not set(X[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]

                if cols_to_scale:
                    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

                
                st.write(f"**Dimensión de datos para el modelo:** {X.shape[0]} pacientes y {X.shape[1]} variables predictoras.")
                
                # Configurar pesos por desbalanceo (ahora y es seguro que tiene 0 y 1)
                count_neg = (y == 0).sum()
                count_pos = (y == 1).sum()

                scale_pos_weight = count_neg / count_pos if count_pos > 0 else 1
                
                # Diccionario de modelo
                model_params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'subsample': subsample,
                    'colsample_bytree': 0.8,
                    'scale_pos_weight': scale_pos_weight,
                    'random_state': 42
                }
                
                # Llamada al módulo (asegúrate de que XGBoost maneje bien los strings si hay category)
                auc_scores, shap_values_list, test_indices_list = ModeloXGBoost.training_skf(
                    n_splits=n_splits, 
                    shuffle=True, 
                    random_state=42, 
                    model=model_params, 
                    X=X, 
                    y=y
                )
                
                auc_mean = np.mean(auc_scores)
                auc_std = np.std(auc_scores)
                
                # Guardar métricas en el historial para comparar
                st.session_state['historial_modelos'].append({
                    'Ejecución': len(st.session_state['historial_modelos']) + 1,
                    'Target': target_col,
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'subsample': subsample,
                    'AUC Promedio': round(auc_mean, 4),
                    'AUC Std': round(auc_std, 4)
                })
                
                # --- RESULTADOS ACTUALES ---
                st.markdown("---")
                st.subheader("📊 Rendimiento de esta Configuración")
                col_met1, col_met2 = st.columns(2)
                col_met1.metric("AUC ROC Promedio", f"{auc_mean:.4f}")
                col_met2.metric("Desviación Estándar (AUC)", f"± {auc_std:.4f}")
                
                st.bar_chart(auc_scores) # Rendimiento por cada fold
                
                # --- INTERPRETABILIDAD CLÍNICA (SHAP) ---
                st.markdown("---")
                st.subheader("🩺 Escala Pronóstica: Importancia de Variables (SHAP)")
                st.markdown("Los gráficos muestran qué peso tiene cada variable clínica en el pronóstico final.")
                
                # Forzar renderizado sin warnings
                st.set_option('deprecation.showPyplotGlobalUse', False)
                # Llamamos a tu función mejorada
                fig_beeswarm, fig_bar = PresentacionClinica.plot_shap_summary(shap_values_list, X, test_indices_list)

                # --- SOLUCIÓN DE UI: CREAR COLUMNAS ---
                col_graf1, col_graf2 = st.columns(2)

                with col_graf1:
                    st.markdown("**Distribución del Impacto (Beeswarm)**")
                    st.pyplot(fig_beeswarm)

                with col_graf2:
                    st.markdown("**Magnitud Absoluta (Barras)**")
                    st.pyplot(fig_bar)

        # --- 4. COMPARATIVA DE MODELOS (HISTORIAL) ---
        if len(st.session_state['historial_modelos']) > 0:
            st.markdown("---")
            st.subheader("📈 Comparativa Histórica de Parámetros")
            st.markdown("Aquí puedes comparar cómo mejoran o empeoran tus resultados al modificar los hiperparámetros:")
            
            # Mostrar como tabla interactiva
            df_historial = pd.DataFrame(st.session_state['historial_modelos'])
            st.dataframe(df_historial, use_container_width=True)
            
            col_btn1, col_btn2 = st.columns([1, 4])
            with col_btn1:
                # Botón para limpiar si hay demasiadas ejecuciones
                if st.button("🧹 Limpiar Historial"):
                    st.session_state['historial_modelos'] = []
                    # st.rerun() obliga a Streamlit a recargar la página para borrar la tabla al instante
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun() # Para versiones antiguas de Streamlit
    else:
        st.warning("⚠️ No hay datos limpios. Por favor, completa la limpieza de datos en la **Pestaña 2** para desbloquear el modelado.")