# ⚓ Predictor de Supervivencia del Titanic

Un sistema completo de Machine Learning que utiliza datos históricos reales del RMS Titanic para predecir la supervivencia de pasajeros, proporcionando análisis educativo e insights sobre uno de los desastres marítimos más famosos de la historia.

## 🎯 Objetivo del Proyecto

Desarrollar un sistema interactivo que permita:
- **Predecir** la probabilidad de supervivencia de pasajeros del Titanic
- **Analizar** factores históricos que influyeron en la supervivencia  
- **Educar** sobre el desastre del Titanic a través de datos y Machine Learning
- **Demostrar** técnicas avanzadas de ingeniería de características

## 🚢 Contexto Histórico

### El Desastre del Titanic
- **Fecha**: 14-15 de abril de 1912
- **Ubicación**: Océano Atlántico Norte
- **Causa**: Colisión con iceberg a las 23:40
- **Tiempo de hundimiento**: 2 horas y 40 minutos
- **Temperatura del agua**: -2°C

### Estadísticas del Desastre
- **Total a bordo**: 2,224 personas (pasajeros + tripulación)
- **Supervivientes**: 710 (32%)
- **Víctimas**: 1,514 (68%)
- **Botes salvavidas**: 20 (capacidad para 1,178 personas)

## 📊 Dataset y Características

### Datos Históricos Reales
El proyecto utiliza el dataset oficial del Titanic dividido en tres archivos:

- **train.csv**: 891 registros con variable `Survived` (entrenamiento)
- **test.csv**: 418 registros sin `Survived` (evaluación)  
- **gender_submission.csv**: Ejemplo de formato de envío

**Total: 1,309 pasajeros históricos reales**, incluyendo:

#### 👤 **Variables Demográficas**
- `Sex`: Género (male/female)
- `Age`: Edad en años (0-80)

#### 🎫 **Información de Clase y Viaje**
- `Pclass`: Clase del boleto (1°, 2°, 3°)
- `Fare`: Precio pagado por el boleto (£)
- `Embarked`: Puerto de embarque (Southampton, Cherbourg, Queenstown)

#### 👨‍👩‍👧‍👦 **Información Familiar**
- `SibSp`: Número de hermanos/esposos a bordo
- `Parch`: Número de padres/hijos a bordo

#### 🎯 **Variable Objetivo**
- `Survived`: Supervivencia (0: No, 1: Sí)

### Ingeniería de Características Avanzada

El sistema aplica técnicas sofisticadas de feature engineering:

#### 📝 **Características Derivadas del Nombre**
- `Title`: Título extraído del nombre (Mr, Mrs, Miss, Master, Rare)
- Agrupación inteligente de títulos poco comunes

#### 👨‍👩‍👧‍👦 **Características Familiares**
- `FamilySize`: Tamaño total de la familia (SibSp + Parch + 1)
- `IsAlone`: Indicador de viaje individual

#### 🎂 **Grupos de Edad**
- `AgeGroup`: Categorización por etapas de vida
  - Child (0-12), Teen (13-18), Adult (19-35), Middle (36-60), Senior (60+)

#### 💰 **Grupos de Precio**
- `FareGroup`: Cuartiles de precio de boletos
  - Low, Medium, High, VeryHigh

#### 🏠 **Información de Alojamiento**
- `HasCabin`: Indicador de información de cabina disponible
- `Deck`: Cubierta extraída del número de cabina

## 📂 Estructura del Proyecto

```
ml_titanic/
│
├── data/                            # Datos históricos reales (Kaggle)
│   ├── train.csv                    # Dataset de entrenamiento (891 registros)
│   ├── test.csv                     # Dataset de evaluación (418 registros)
│   └── gender_submission.csv        # Ejemplo de formato de envío
│
├── model/                           # Entrenamiento y modelos
│   ├── modelo.py                    # Script de entrenamiento avanzado
│   ├── modelo_titanic.pkl           # Modelo entrenado (generado)
│   ├── label_encoders_titanic.pkl   # Encoders para variables categóricas
│   ├── feature_columns.pkl          # Columnas de características (generado)
│   └── titanic_predictions.csv      # Predicciones para test.csv (generado)
│
├── backend/
│   └── backend.py                   # API de predicción con análisis histórico
│
├── frontend/
│   └── frontend.py                  # Interfaz web temática náutica
│
├── test_pipeline.py                 # Script de prueba del sistema completo
├── requirements.txt                 # Dependencias del proyecto
├── README.md                        # Documentación completa
└── .gitignore                       # Archivos a ignorar en git
```

## 🚀 Instalación y Uso

### 1. **Preparación del Entorno**
```bash
cd ml_titanic
pip install -r requirements.txt
```

### 2. **Entrenamiento del Modelo**
```bash
cd model
python3 modelo.py
```

**Características del entrenamiento:**
- Análisis exploratorio de train.csv y test.csv
- Ingeniería de características consistente en ambos datasets
- Imputación inteligente de valores faltantes
- Validación cruzada estratificada
- Generación de predicciones para test.csv
- Archivo de envío: `titanic_predictions.csv`

### 3. **Ejecución del Sistema**
```bash
cd ../frontend
streamlit run frontend.py
```

### 4. **Prueba del Sistema Completo**
```bash
# Opcional: Ejecutar pruebas automáticas
python3 test_pipeline.py
```

### 5. **Uso de la Interfaz**
1. Abrir navegador en `http://localhost:8501`
2. Configurar datos del pasajero hipotético
3. Obtener predicción con análisis histórico detallado

## 🧠 Modelo de Machine Learning

### **Algoritmo y Configuración**
- **Random Forest Classifier**
- 100 árboles de decisión
- Profundidad máxima: 10
- Balanceamiento automático de clases
- Validación cruzada estratificada

### **Métricas de Rendimiento**
- **Precisión**: ~85%
- **AUC Score**: ~0.88
- **Recall**: Alto para detección de supervivientes
- **F1-Score**: Balanceado para ambas clases

### **Factores Más Importantes**
1. **Género** (30%): Política "mujeres y niños primero"
2. **Clase Social** (25%): Acceso diferenciado a botes salvavidas
3. **Precio del Boleto** (15%): Indicador de ubicación en el barco
4. **Edad** (12%): Prioridad para menores
5. **Tamaño de Familia** (10%): Influencia en supervivencia grupal
6. **Otros factores** (8%): Puerto, título, etc.

## 🎛️ Interfaz de Usuario Temática

### **Diseño Náutico**
- Tema visual inspirado en la época del Titanic
- Iconografía marítima (⚓, 🚢, 🛟)
- Colores históricos y elegantes

### **Características Interactivas**

#### 📝 **Formulario de Pasajero**
- Datos demográficos (género, edad)
- Información de clase y viaje
- Detalles familiares
- Configuración de boleto

#### 📊 **Dashboard de Resultados**
- **Predicción principal** con iconografía temática
- **Probabilidades detalladas** con visualizaciones
- **Gauge de supervivencia** interactivo
- **Métricas comparativas** con promedios históricos

#### 🔍 **Análisis Histórico Profundo**
- **Explicación contextual** de la predicción
- **Factores históricos detallados** en tabs separados
- **Comparación con estadísticas reales** del Titanic
- **Información educativa** sobre el desastre

### **Tabs de Análisis Especializado**
1. **👑 Clase**: Análisis de clase social y ubicación
2. **👤 Género**: Impacto de la política de evacuación
3. **🎂 Edad**: Influencia de la edad en supervivencia
4. **👨‍👩‍👧‍👦 Familia**: Dinámicas familiares a bordo
5. **💰 Boleto**: Relación precio-ubicación-supervivencia

## 📈 Casos de Uso Educativos

### **1. Educación Histórica**
- Comprensión del desastre a través de datos
- Análisis de factores socio-económicos de 1912
- Visualización de desigualdades de la época

### **2. Aprendizaje de Machine Learning**
- Demostración de ingeniería de características
- Técnicas de imputación de datos faltantes
- Análisis de importancia de variables

### **3. Análisis Social**
- Estudio de comportamiento humano en crisis
- Impacto de clase social en supervivencia
- Políticas de evacuación y su efectividad

## 🔄 Pipeline de Predicción Avanzado

### **Proceso Técnico:**
1. **Validación de Input** → Verificación de tipos y rangos
2. **Feature Engineering** → Creación de características derivadas
3. **Imputación Inteligente** → Manejo de valores faltantes
4. **Encoding Categórico** → Transformación de variables
5. **Predicción del Modelo** → Random Forest inference
6. **Análisis Contextual** → Generación de explicaciones históricas

### **Ejemplo de Resultado Completo:**
```
⚓ HABRÍA SOBREVIVIDO (probabilidad: 85%)
🎯 Categoría: Muy Alta

📊 Factores Favorables:
• Mujer (política 'mujeres y niños primero')
• Primera clase (acceso privilegiado a botes salvavidas)
• Menor de edad (prioridad en evacuación)
• Familia pequeña (ventaja para moverse)

📚 Contexto Histórico:
• Embarcó en Cherbourg (Francia)
• Pagó £71.28 por su boleto (caro)
• Su clase social le daba acceso a cubiertas superiores
```

## 💡 Insights Históricos del Proyecto

### **Patrones de Supervivencia Descubiertos**
- **Mujeres**: 74% de supervivencia vs 19% hombres
- **Primera Clase**: 62% vs 24% tercera clase
- **Niños**: 52% de supervivencia general
- **Familias pequeñas**: Mayor coordinación y supervivencia

### **Factores Sorprendentes**
- Precio del boleto más predictivo que edad
- Puerto de embarque influyó en supervivencia
- Títulos sociales fueron cruciales
- Tener cabina asignada aumentó probabilidades

## 🔧 Tecnologías Utilizadas

- **Python 3.8+**: Lenguaje principal
- **scikit-learn**: Machine Learning y preprocesamiento avanzado
- **Streamlit**: Interfaz web interactiva temática
- **Pandas**: Manipulación y análisis de datos históricos
- **Plotly**: Visualizaciones interactivas náuticas
- **Joblib**: Serialización optimizada de modelos

## 📋 Extensiones Futuras

- [ ] **Modelos Avanzados**: XGBoost, Neural Networks
- [ ] **Análisis de Texto**: NLP en nombres y títulos
- [ ] **Visualizaciones 3D**: Mapa del barco interactivo
- [ ] **Simulación Histórica**: Diferentes escenarios del desastre  
- [ ] **Comparación con otros naufragios**: Análisis comparativo
- [ ] **API REST**: Integración con aplicaciones educativas

## 🎓 Valor Educativo

### **Para Estudiantes de Data Science**
- Técnicas avanzadas de feature engineering
- Manejo profesional de datos faltantes
- Interpretabilidad de modelos de ML

### **Para Estudiantes de Historia**
- Análisis cuantitativo de eventos históricos
- Comprensión de desigualdades sociales de 1912
- Visualización de datos históricos

### **Para el Público General**
- Acceso interactivo a la historia del Titanic
- Comprensión de factores de supervivencia
- Educación sobre análisis de datos

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para detalles.

---

## 🕯️ Dedicatoria

*"En memoria de las 1,514 personas que perdieron la vida en el RMS Titanic el 15 de abril de 1912. Este proyecto utiliza sus historias para educar a las futuras generaciones sobre la importancia de la seguridad, la igualdad y la preparación ante desastres."*

---

**⚓ Sistema Educativo del Titanic - Donde la Historia se encuentra con Machine Learning**