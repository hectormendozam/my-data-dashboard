import io
import pandas as pd
import numpy as np
from django.views.generic import TemplateView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.urls import reverse # Importar reverse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.views import APIView # Asegúrate de que esta esté importada


import random # Se mantiene solo para añadir un ligero factor de variabilidad en las métricas heurísticas

# ==============================================================================
# 1. FUNCIÓN DE ANÁLISIS DE DATASET (LA FUNCIÓN REAL)
# ==============================================================================
def analyze_dataset(file_obj, filename="dataset.csv"):
    """
    Lee un archivo CSV (en un FileObject), realiza un análisis de calidad de datos 
    y devuelve las métricas en el formato JSON esperado por el frontend.
    """
    
    # --- Lectura y Preparación de Datos ---
    # Decodificar el archivo en memoria para que Pandas lo pueda leer
    data = file_obj.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(data))
    
    total_rows = len(df)
    total_columns = len(df.columns)
    
    # --- 1. Información Básica ---
    
    # Conteo de tipos de datos
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    
    basic_info = {
        "total_rows": total_rows,
        "total_columns": total_columns,
        # Tamaño del archivo en MB, redondeado a 2 decimales
        "file_size": f"{file_obj.size / (1024*1024):.2f} MB", 
        "data_types": dtype_counts,
    }
    
    # --- 2. Valores Faltantes ---
    
    missing_data_series = df.isnull().sum()
    total_missing_count = missing_data_series.sum()
    
    # Porcentaje total sobre el dataset completo (Filas * Columnas)
    total_data_points = total_rows * total_columns
    total_missing_percentage = (total_missing_count / total_data_points) * 100 if total_data_points > 0 else 0
    
    columns_with_missing = []
    # Filtrar solo columnas con valores faltantes y ordenar
    for col, count in missing_data_series[missing_data_series > 0].sort_values(ascending=False).items():
        percent = (count / total_rows) * 100
        columns_with_missing.append({"column": col, "percentage": round(percent, 2)})
        
    missing_data = {
        "total_missing_percentage": round(total_missing_percentage, 2),
        "columns_with_missing": columns_with_missing,
    }

    # --- 3. Duplicados ---
    
    total_duplicates = df.duplicated().sum()
    duplicate_percentage = (total_duplicates / total_rows) * 100 if total_rows > 0 else 0
    
    # Identificar columnas que contribuyen a duplicados (simplificado: 
    # columnas que no tienen valores únicos para casi todas las filas)
    columns_contributing = df.apply(lambda x: x.duplicated().sum() > 0).loc[lambda x: x].index.tolist()
    columns_contributing = columns_contributing[:5] # Limitar a las 5 principales
    
    duplicates = {
        "total_duplicates": int(total_duplicates),
        "percentage": round(duplicate_percentage, 2),
        "columns_contributing": columns_contributing,
    }

    # --- 4. Métricas de Calidad (Heurísticas Basadas en Análisis) ---
    
    # Completeness (Inverso del porcentaje total de missing en el dataset)
    completeness = 100.0 - total_missing_percentage
    
    # Uniqueness (Inverso del porcentaje de filas duplicadas)
    uniqueness = 100.0 - duplicate_percentage
    
    # Consistency Proxy (Mide la variabilidad; alta variabilidad = alta consistencia. Simplificado)
    # Se castiga si hay muchas columnas con una sola categoría única (baja variabilidad/información)
    consistency_score = 100.0 - (df.nunique().value_counts().get(1, 0) / total_columns * 100)
    consistency = max(0, min(100, consistency_score)) 
    
    # Validity Proxy (Mide cuántos puntos numéricos caen fuera del rango de 3 desviaciones estándar)
    numeric_cols = df.select_dtypes(include=np.number).columns
    valid_data_points = 0
    total_numeric_points = 0
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        mean = col_data.mean()
        std = col_data.std()
        # Definir límites de 3-sigma
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        
        valid_count = ((col_data >= lower_bound) & (col_data <= upper_bound)).sum()
        valid_data_points += valid_count
        total_numeric_points += len(col_data)

    validity = (valid_data_points / total_numeric_points) * 100 if total_numeric_points > 0 else 100.0
    validity = max(0, min(100, validity)) # Asegurar que esté entre 0 y 100
    
    data_quality = {
        "completeness": round(completeness, 2),
        "consistency": round(consistency, 2),
        "validity": round(validity, 2),
        "uniqueness": round(uniqueness, 2),
    }

    # --- 5. Outliers (Método IQR) ---
    outliers_data = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Conteo de valores que caen fuera de los límites
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if outlier_count > 0:
            outliers_data.append({"column": col, "outlier_count": int(outlier_count)})
    
    outliers_data.sort(key=lambda x: x['outlier_count'], reverse=True)
    outliers = {"columns_with_outliers": outliers_data}

    # --- 6. Correlación (Top 5 Absolutas) ---
    correlation_data = []
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().unstack()
        
        unique_correlations = set()
        for pair, value in corr_matrix.items():
            var1, var2 = pair
            # Filtrar auto-correlaciones y duplicados (A vs B y B vs A)
            if var1 != var2 and (var2, var1) not in unique_correlations and not pd.isna(value):
                correlation_data.append({
                    "var1": var1,
                    "var2": var2,
                    "correlation": round(value, 4)
                })
                unique_correlations.add((var1, var2))

        # Ordenar por el valor absoluto y tomar los 5 principales
        correlation_data.sort(key=lambda x: abs(x['correlation']), reverse=True)
        # ⚠️ Esta variable ahora se está reasignando, pero ya existía si el IF es falso.
        correlation_matrix = correlation_data[:5] 
    else:
        correlation_matrix = []
    
    
    # --- 7. Recomendaciones (Basadas en Umbrales) ---
    recommendations = {
        "critical": [],
        "moderate": [],
        "optional": []
    }
    
    if total_missing_percentage > 10:
        recommendations['critical'].append({"description": f"El dataset tiene {total_missing_percentage:.2f}% de datos faltantes, lo cual es Crítico. Se requiere una estrategia de imputación o eliminación."})
    if duplicate_percentage > 5:
        recommendations['critical'].append({"description": f"El {duplicate_percentage:.2f}% de las filas son duplicados. Se recomienda la eliminación de duplicados antes del análisis."})
    
    if total_missing_percentage > 2 and total_missing_percentage <= 10:
        recommendations['moderate'].append({"description": f"El {total_missing_percentage:.2f}% de datos faltantes requiere imputación o revisión."})
    if len(outliers_data) > 0:
        recommendations['moderate'].append({"description": f"Se identificaron outliers significativos en {len(outliers_data)} columnas numéricas. Considere el tratamiento de valores atípicos."})
    
    recommendations['optional'].append({"description": "Normalización/escalado de variables numéricas para el entrenamiento de modelos."})
    if len(df.select_dtypes(include=['object', 'category']).columns) > 5:
        recommendations['optional'].append({"description": "Encoding de variables categóricas (One-Hot Encoding, Label Encoding) para el uso en algoritmos."})
    
    
    # --- 8. Retorno Final ---
    return {
        "analysis": {
            "basic_info": basic_info,
            "data_quality": data_quality,
            "missing_data": missing_data,
            "duplicates": duplicates,
            "outliers": outliers,
            "correlation_matrix": correlation_matrix,
            "recommendations": recommendations,
        }
    }


# ==============================================================================
# 2. CLASES DE VISTA DE DJANGO (DRF)
# ==============================================================================

class DashboardView(TemplateView):
    """Renderiza la plantilla principal del dashboard."""
    template_name = "dashboard/dashboard.html"

    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        context['api_upload_url'] = reverse('api_upload')
        
        return context
    
@method_decorator(csrf_exempt, name='dispatch')

class DatasetUploadView(APIView):
    """Maneja la subida del archivo y ejecuta el análisis real."""
    
    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        
        if not file_obj:
            return Response({'error': 'No se proporcionó archivo'}, status=status.HTTP_400_BAD_REQUEST)

        filename = file_obj.name
        
        if not filename.lower().endswith('.csv'):
            return Response({'error': 'Solo se admiten archivos CSV'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Llama a la función de análisis REAL
            analysis_result = analyze_dataset(file_obj, filename) 
            
            return Response(analysis_result, status=status.HTTP_200_OK)

        except Exception as e:
            # Captura errores durante el análisis (ej. CSV malformado)
            print(f"Error fatal durante el análisis: {e}")
            return Response(
                {'error': f'Error al procesar el archivo. Verifique el formato CSV: {e}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )