#!/usr/bin/env bash

# Salir inmediatamente si un comando falla
set -o errexit

# 1. Instalar dependencias
pip install -r requirements.txt

# 2. EJECUTAR MIGRACIONES
# Nota: La base de datos debe estar configurada en settings.py para este punto.
python manage.py makemigrations dashboard
python manage.py migrate --no-input

# 3. Recolectar archivos estáticos
python manage.py collectstatic --no-input

# ¡Ya no necesitas el shell de Render para esto!