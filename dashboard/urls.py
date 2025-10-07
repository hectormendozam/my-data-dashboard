from django.urls import path
from .views import DatasetUploadView, DashboardView 

urlpatterns = [
    # 1. Ruta de la Interfaz (Frontend): /
    path('', DashboardView.as_view(), name='dashboard'), 
    
    # 2. Ruta de la API (Backend): /api/upload/
    path('api/upload/', DatasetUploadView.as_view(), name='api_upload'),
]