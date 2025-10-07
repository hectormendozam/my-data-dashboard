#!/usr/bin/env bash

# Salir inmediatamente si un comando falla
set -o errexit

# Instalar dependencias
pip install -r requirements.txt

# Recolectar archivos estáticos
python manage.py collectstatic --no-input