import pandas as pd
import numpy as np


def leer_datos(nombre_archivo):
    """
    Lee un archivo SAV y devuelve un DataFrame de pandas.

    Args:
        nombre_archivo (str): El nombre del archivo SAV a leer.

    Returns:
        pd.DataFrame: Un DataFrame que contiene los datos del archivo SAV.
    """
    try:
        datos = pd.read_spss(nombre_archivo)
        return datos
    except FileNotFoundError:
        print(f"Error: El archivo '{nombre_archivo}' no se encontró.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: El archivo '{nombre_archivo}' está vacío.")
        return None
    except pd.errors.ParserError:
        print(f"Error: El archivo '{nombre_archivo}' no se pudo parsear correctamente.")
        return None

def analizar_datos(datos):
    """
    Realiza un análisis básico de los datos.

    Args:
        datos (pd.DataFrame): El DataFrame que contiene los datos a analizar.

    Returns:
        dict: Un diccionario con estadísticas básicas de los datos.
    """
    if datos is None:
        print("No se pueden analizar los datos porque el DataFrame es None.")
        return None

    estadisticas = {
        "num_filas": len(datos),
        "num_columnas": len(datos.columns),
        "columnas": list(datos.columns),
        "tipos_datos": datos.dtypes.to_dict(),
        "resumen_estadistico": datos.describe(include='all').to_dict(),
        "valores_nulos": datos.isnull().sum().to_dict(),
        "valores_unicos": {col: int(datos[col].nunique()) for col in datos.columns},
        "valores_nulos_por_columna": {col: int(datos[col].isnull().sum()) for col in datos.columns}
    }
    return estadisticas

# REGLA 1: LIMPIAR SEGUN EL INTERES
def limpieza_datos(datos, threshold: float = 0.2, cols_to_drop_manual: list  = None, cols_to_save: list = None):
    """
    Realiza una limpieza básica de los datos, elimina filas con más de un 20% de valores nulos.
    Las columnas en cols_to_save se preservan sin aplicar el filtro de datos faltantes.

    Args:
        datos (pd.DataFrame): El DataFrame que contiene los datos a limpiar.
        cols_to_drop_manual (list, optional): Lista de columnas a eliminar manualmente.
        cols_to_save (list, optional): Lista de columnas a conservar (se preservan aunque tengan muchos datos faltantes).
    Returns:
        pd.DataFrame: El DataFrame limpio.
    """
    datos_limpio = datos.copy()
    
    # Eliminar columnas especificadas manualmente
    if cols_to_drop_manual is not None:
        datos_limpio = datos_limpio.drop(columns=cols_to_drop_manual, errors='ignore')
    
    # Aplicar filtro de datos faltantes
    umbral = 0.2 * len(datos_limpio.columns)
    datos_limpio = datos_limpio.dropna(thresh=umbral)
    
    # Si se especifica cols_to_save, asegurar que estén incluidas
    if cols_to_save is not None:
        columnas_disponibles = list(datos_limpio.columns)
        columnas_no_encontradas = set(cols_to_save) - set(datos.columns)
        
        # Validar columnas en el DataFrame original
        if not any(col in datos.columns for col in cols_to_save):
            print("Advertencia: Ninguna columna en 'cols_to_save' se encuentra en el DataFrame. Se ignorarán esas columnas.")
        else:
            if columnas_no_encontradas:
                print(f"Advertencia: Las siguientes columnas no se encuentran en el DataFrame: {columnas_no_encontradas}")
            
            # Agregar las columnas a guardar aunque no pasen el filtro de datos faltantes
            cols_validas = [col for col in cols_to_save if col in datos.columns]
            
            # Combinar columnas: las que pasaron el filtro + las que debemos guardar sí o sí
            cols_finales = list(set(columnas_disponibles) | set(cols_validas))
            datos_limpio = datos[cols_finales]
    
    return datos_limpio


def missing_as_information(datos, col_name: str):
    """
    Transforma los valores faltantes de una columna en una nueva categoría.

    Args:
        datos (pd.DataFrame): El DataFrame que contiene los datos a transformar.
        col_name (str): El nombre de la columna a transformar.

    Returns:
        pd.DataFrame: El DataFrame con la columna transformada.
    """
    if col_name not in datos.columns:
        print(f"Error: La columna '{col_name}' no se encuentra en el DataFrame.")
        return datos

    datos[f"{col_name}_missing"] = datos[col_name].isnull().astype(categorical_dtype := pd.api.types.CategoricalDtype(categories=[False, True], ordered=True))
    return datos




if __name__ == "__main__":
    nombre_archivo = "datos.sav"  # Reemplaza con el nombre de tu archivo SAV
    datos = leer_datos(nombre_archivo)
    if datos is not None:
        print("Nombres de las columnas:")
        print(datos.head())  # Muestra las primeras filas del DataFrame  
        datos_limpios = missing_as_information(datos, 'SARCF')
        datos_limpios = limpieza_datos(datos_limpios, threshold=0.05,
                                       cols_to_drop_manual=['NÚM', 'FECHA_ING_CORREGIDA', 'FECHA_ALTA',
                                                            'MORTALIDAD_FECHA', 'CONSENTIMIENTO_FIRMADO'], 
                                       cols_to_save=['MORTALIDAD']
                                       )

        estadisticas = analizar_datos(datos_limpios)
        print("\nEstadísticas generales:")
        for key, value in estadisticas.items():
            if isinstance(value, list):
                print(f"{key}: {len(value)} elementos")
            elif isinstance(value, dict):
                print(f"{key}: {len(value)} elementos")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")


        print("\n Get SARCF missing values as information:")
        print(datos_limpios['SARCF_missing'].value_counts(dropna=False))