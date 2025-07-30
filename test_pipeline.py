#!/usr/bin/env python3
"""
Script de prueba para verificar el pipeline completo del Titanic
"""

import os
import sys

def test_data_files():
    """Verificar que los archivos de datos estén presentes"""
    print("🔍 Verificando archivos de datos...")
    
    data_files = [
        'data/train.csv',
        'data/test.csv', 
        'data/gender_submission.csv'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            lines = sum(1 for line in open(file_path))
            print(f"  ✅ {file_path}: {lines} líneas")
        else:
            print(f"  ❌ {file_path}: No encontrado")
            return False
    
    return True

def test_model_training():
    """Probar el entrenamiento del modelo"""
    print("\n🧠 Probando entrenamiento del modelo...")
    
    # Cambiar al directorio del modelo
    original_dir = os.getcwd()
    os.chdir('model')
    
    try:
        # Importar y ejecutar el módulo
        sys.path.append('.')
        import modelo
        
        print("  🚀 Ejecutando entrenamiento...")
        modelo.main()
        
        # Verificar archivos generados
        generated_files = [
            'modelo_titanic.pkl',
            'label_encoders_titanic.pkl',
            'feature_columns.pkl',
            'titanic_predictions.csv'
        ]
        
        all_generated = True
        for file_path in generated_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"  ✅ {file_path}: {size} bytes")
            else:
                print(f"  ❌ {file_path}: No generado")
                all_generated = False
        
        return all_generated
        
    except Exception as e:
        print(f"  ❌ Error durante entrenamiento: {str(e)}")
        return False
    finally:
        os.chdir(original_dir)

def test_backend():
    """Probar el backend de predicción"""
    print("\n⚙️ Probando backend de predicción...")
    
    try:
        # Cambiar al directorio del backend
        original_dir = os.getcwd()
        os.chdir('backend')
        
        sys.path.append('.')
        from backend import TitanicSurvivalPredictor
        
        # Inicializar predictor
        predictor = TitanicSurvivalPredictor()
        
        # Datos de prueba
        test_data = {
            'Pclass': 1,
            'Sex': 'female',
            'Age': 38,
            'SibSp': 1,
            'Parch': 0,
            'Fare': 71.28,
            'Embarked': 'C'
        }
        
        # Realizar predicción
        result = predictor.predict_with_explanation(test_data)
        
        if 'error' not in result:
            print(f"  ✅ Predicción exitosa: {result['prediction_label']}")
            print(f"  📊 Probabilidad: {result['survival_probability']:.1%}")
            return True
        else:
            print(f"  ❌ Error en predicción: {result['error']}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error en backend: {str(e)}")
        return False
    finally:
        os.chdir(original_dir)

def test_frontend_syntax():
    """Verificar sintaxis del frontend"""
    print("\n🎨 Verificando sintaxis del frontend...")
    
    try:
        import py_compile
        py_compile.compile('frontend/frontend.py', doraise=True)
        print("  ✅ Sintaxis del frontend correcta")
        return True
    except py_compile.PyCompileError as e:
        print(f"  ❌ Error de sintaxis en frontend: {str(e)}")
        return False

def main():
    """Ejecutar todas las pruebas"""
    print("🚢 PRUEBA COMPLETA DEL PIPELINE DEL TITANIC")
    print("=" * 50)
    
    tests = [
        ("Archivos de datos", test_data_files),
        ("Entrenamiento del modelo", test_model_training),
        ("Backend de predicción", test_backend),
        ("Sintaxis del frontend", test_frontend_syntax)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ Error inesperado en {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Resumen de resultados
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Resultado: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("🚀 El sistema está listo para usar:")
        print("   1. cd model && python3 modelo.py")
        print("   2. cd ../frontend && streamlit run frontend.py")
    else:
        print(f"\n⚠️  {total - passed} pruebas fallaron")
        print("🔧 Revisa los errores anteriores antes de usar el sistema")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)