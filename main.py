import streamlit as st
import pandas as pd
import os
import io

# Importar tus módulos
import LecturaDatos
import ModeloXGBoost
import PresentacionClinica

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
tab1, tab2, tab3 = st.tabs(["📁 1. Carga de Datos", "⚙️ 2. Preprocesamiento", "🚀 3. Modelado Predictivo"])

# ==========================================
# PESTAÑA 1: CARGA DE DATOS
# ==========================================

with tab1:
    st.header("Carga de Archivo de Datos")
    uploaded_file = st.file_uploader("Sube tu archivo de datos (.sav o .csv)", type=["sav", "csv"])
    
    if uploaded_file is not None:
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

        # Mostrar resultados
        if st.session_state['df_raw'] is not None:
            st.success(f"Archivo cargado exitosamente. ({st.session_state['df_raw'].shape[0]} filas, {st.session_state['df_raw'].shape[1]} columnas)")
            st.dataframe(st.session_state['df_raw'].head())

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
            # Mostrar tabla de resumen de nulos
            resumen_nulos = pd.DataFrame({
                'Tipo': df.dtypes,
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
            
            # Las columnas disponibles para guardar son las que no se van a eliminar
            cols_disponibles_save = [c for c in columnas_disponibles if c not in cols_to_drop]
            cols_to_save = st.multiselect(
                "Columnas a GUARDAR (ignorar threshold) (cols_to_save)", 
                options=cols_disponibles_save,
                help="Ejemplo: Tu variable objetivo 'MORTALIDAD'. No se eliminará aunque tenga muchos nulos."
            )
            
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
            st.divider()
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
# PESTAÑA 3: MODELADO PREDICTIVO (En construcción)
# ==========================================
with tab3:
    st.header("Entrenamiento y Evaluación del Modelo")
    if st.session_state['df_clean'] is not None:
        st.info("Aquí irá el código para seleccionar la variable objetivo dinámica, los hiperparámetros de XGBoost y las gráficas SHAP.")
        # Aquí trabajaremos en el Paso 2...
    else:
        st.warning("Debes completar la limpieza de datos en la Pestaña 2 para acceder al modelado.")