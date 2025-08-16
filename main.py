#!/usr/bin/env python3
"""
Script principal para inicializar la aplicación Titanic ML
- Inicia la aplicación Streamlit directamente
"""

import sys
import subprocess

def start_streamlit():
    """Inicia la aplicación Streamlit"""
    print("🚀 Iniciando aplicación Streamlit...")
    try:
        subprocess.run([
            'streamlit', 'run', 'frontend/frontend.py',
            '--server.address=0.0.0.0',
            '--server.port=8501'
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Aplicación detenida por el usuario")
    except Exception as e:
        print(f"❌ Error al iniciar Streamlit: {e}")
        sys.exit(1)

def main():
    """Función principal"""
    print("🚢 Iniciando aplicación Titanic ML...")
    
    # Iniciar Streamlit
    start_streamlit()

if __name__ == "__main__":
    main()