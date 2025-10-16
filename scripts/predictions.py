import pandas as pd

def ordenar_y_guardar(df: pd.DataFrame, col_sort: str, col_output: str, ascending: bool = True):
    """
    Ordena un DataFrame por una columna y guarda otra columna en un archivo llamado 'prediction.csv'.
    
    Parámetros:
    - df: DataFrame de entrada
    - col_sort: nombre de la columna por la que se ordenará
    - col_output: nombre de la columna que se guardará
    - ascending: True para orden ascendente, False para descendente
    """
    # Verificar que las columnas existen
    if col_sort not in df.columns or col_output not in df.columns:
        raise ValueError("Una o ambas columnas especificadas no existen en el DataFrame.")
    
    # Ordenar el DataFrame
    df_sorted = df.sort_values(by=col_sort, ascending=ascending)
    
    # Seleccionar solo la columna de salida
    output_series = df_sorted[col_output]
    
    # Guardar en archivo
    output_series.to_csv("prediction.csv", index=False)
    
    print(f"Archivo 'prediction.csv' guardado con {len(output_series)} registros.")
    return output_series
