#!/bin/bash

# Streamlit startup script for Azure App Service
python -m streamlit run streamlit_from_csv.py --server.port 8000 --server.address 0.0.0.0
