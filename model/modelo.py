import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import os

def load_and_analyze_data(file_path):
    """Carga y analiza el dataset del Titanic"""
    print("Cargando dataset del Titanic...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset cargado con éxito: {df.shape[0]} registros, {df.shape[1]} columnas")
    print("\nPrimeras 5 filas:")
    print(df.head())
    
    print("\nInformación del dataset:")
    print(df.info())
    
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    
    print("\nDistribución de supervivencia:")
    if 'Survived' in df.columns:
        print(df['Survived'].value_counts())
        print(f"Tasa de supervivencia: {df['Survived'].mean():.2%}")
    
    print("\nDistribución por clase:")
    print(df['Pclass'].value_counts().sort_index())
    
    print("\nDistribución por sexo:")
    print(df['Sex'].value_counts())
    
    print("\nDistribución por puerto de embarque:")
    print(df['Embarked'].value_counts())
    
    return df

def feature_engineering(df):
    """Ingeniería de características para el dataset del Titanic"""
    print("\nIniciando ingeniería de características...")
    
    # Crear una copia para no modificar el original
    df_processed = df.copy()
    
    # 1. Crear característica 'Title' a partir del nombre
    df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    
    # Agrupar títulos poco comunes
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    df_processed['Title'] = df_processed['Title'].map(title_mapping)
    df_processed['Title'].fillna('Rare', inplace=True)
    
    print("Distribución de títulos:")
    print(df_processed['Title'].value_counts())
    
    # 2. Crear característica 'FamilySize'
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    
    # 3. Crear característica 'IsAlone'
    df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
    
    # 4. Procesar Age - rellenar valores nulos con la mediana por título
    for title in df_processed['Title'].unique():
        if pd.notna(title):
            median_age = df_processed[df_processed['Title'] == title]['Age'].median()
            df_processed.loc[(df_processed['Age'].isnull()) & (df_processed['Title'] == title), 'Age'] = median_age
    
    # Si aún quedan nulos, usar la mediana general
    df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
    
    # 5. Crear grupos de edad
    df_processed['AgeGroup'] = pd.cut(df_processed['Age'], 
                                    bins=[0, 12, 18, 35, 60, 80], 
                                    labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # 6. Procesar Fare
    df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)
    
    # Crear grupos de fare
    df_processed['FareGroup'] = pd.qcut(df_processed['Fare'], 
                                       q=4, 
                                       labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # 7. Procesar Embarked
    df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
    
    # 8. Crear característica de cabina
    df_processed['HasCabin'] = (~df_processed['Cabin'].isnull()).astype(int)
    
    # 9. Extraer deck de la cabina
    df_processed['Deck'] = df_processed['Cabin'].str[0]
    df_processed['Deck'].fillna('Unknown', inplace=True)
    
    print(f"\nCaracterísticas creadas:")
    print(f"- Title: {df_processed['Title'].nunique()} categorías únicas")
    print(f"- FamilySize: rango {df_processed['FamilySize'].min()}-{df_processed['FamilySize'].max()}")
    print(f"- IsAlone: {df_processed['IsAlone'].sum()} pasajeros solos")
    print(f"- AgeGroup: {df_processed['AgeGroup'].value_counts().to_dict()}")
    print(f"- HasCabin: {df_processed['HasCabin'].sum()} con información de cabina")
    
    return df_processed

def preprocess_data(df):
    """Preprocesamiento de datos para el modelo"""
    print("\nIniciando preprocesamiento...")
    
    # Seleccionar características para el modelo
    feature_columns = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
        'Title', 'FamilySize', 'IsAlone', 'AgeGroup', 'FareGroup', 'HasCabin', 'Deck'
    ]
    
    # Crear DataFrame con características seleccionadas
    if 'Survived' in df.columns:
        df_model = df[feature_columns + ['Survived']].copy()
    else:
        df_model = df[feature_columns].copy()
    
    # Codificar variables categóricas
    label_encoders = {}
    categorical_columns = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'Deck']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le
        print(f"Codificada columna '{col}': {list(le.classes_)}")
    
    # Separar características (X) y variable objetivo (y) si existe
    if 'Survived' in df_model.columns:
        X = df_model.drop(['Survived'], axis=1)
        y = df_model['Survived']
        
        print(f"\nCaracterísticas (X): {X.shape}")
        print(f"Variable objetivo (y): {y.shape}")
        print(f"Distribución de supervivencia:")
        print(y.value_counts())
        
        return X, y, label_encoders, feature_columns
    else:
        X = df_model
        print(f"\nCaracterísticas (X): {X.shape}")
        return X, None, label_encoders, feature_columns

def split_data(X, y, test_size=0.2, random_state=42):
    """Divide los datos en entrenamiento y prueba"""
    print(f"\nDividiendo datos: {int((1-test_size)*100)}% entrenamiento, {int(test_size*100)}% prueba")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Entrenamiento: {X_train.shape[0]} registros")
    print(f"Prueba: {X_test.shape[0]} registros")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Entrena el modelo Random Forest para predicción de supervivencia"""
    print("\nEntrenando modelo Random Forest para predicción de supervivencia...")
    
    # Crear y entrenar el modelo
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("Modelo entrenado exitosamente!")
    
    # Mostrar importancia de características
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nImportancia de características:")
    print(feature_importance)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evalúa el rendimiento del modelo"""
    print("\nEvaluando modelo...")
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Precisión: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=['No Sobrevivió', 'Sobrevivió']))
    
    print("\nMatriz de confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Análisis adicional
    print(f"\nAnálisis de supervivencia:")
    print(f"Verdaderos Negativos (No supervivencia correcta): {cm[0,0]}")
    print(f"Falsos Positivos (Predijo supervivencia, pero no): {cm[0,1]}")
    print(f"Falsos Negativos (No predijo supervivencia, pero sí): {cm[1,0]}")
    print(f"Verdaderos Positivos (Supervivencia correcta): {cm[1,1]}")
    
    return accuracy, auc_score

def save_model_and_encoders(model, label_encoders, feature_columns, model_path='modelo_titanic.pkl', 
                           encoders_path='label_encoders_titanic.pkl', features_path='feature_columns.pkl'):
    """Guarda el modelo, encoders y columnas de características"""
    print(f"\nGuardando modelo en {model_path}...")
    joblib.dump(model, model_path)
    
    print(f"Guardando encoders en {encoders_path}...")
    joblib.dump(label_encoders, encoders_path)
    
    print(f"Guardando columnas de características en {features_path}...")
    joblib.dump(feature_columns, features_path)
    
    print("Modelo, encoders y características guardados exitosamente!")

def main():
    """Función principal que ejecuta todo el pipeline del Titanic"""
    # Rutas de archivos
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    model_path = 'modelo_titanic.pkl'
    encoders_path = 'label_encoders_titanic.pkl'
    features_path = 'feature_columns.pkl'
    
    # Verificar que los archivos de datos existen
    if not os.path.exists(train_path):
        print(f"Error: No se encontró el archivo {train_path}")
        return
    
    if not os.path.exists(test_path):
        print(f"Error: No se encontró el archivo {test_path}")
        return
    
    try:
        # 1. Cargar y analizar datos de entrenamiento
        print("="*70)
        print("CARGANDO DATOS DE ENTRENAMIENTO")
        print("="*70)
        df_train = load_and_analyze_data(train_path)
        
        # 2. Cargar datos de prueba (para verificar consistencia)
        print("\n" + "="*70)
        print("CARGANDO DATOS DE PRUEBA (para verificación)")
        print("="*70)
        df_test = load_and_analyze_data(test_path)
        
        # 3. Combinar datasets para ingeniería de características consistente
        print("\n" + "="*70)
        print("COMBINANDO DATASETS PARA FEATURE ENGINEERING")
        print("="*70)
        # Marcar origen de los datos
        df_train['dataset'] = 'train'
        df_test['dataset'] = 'test'
        
        # Combinar temporalmente (test no tiene Survived)
        df_combined = pd.concat([df_train, df_test], sort=False, ignore_index=True)
        print(f"Dataset combinado: {df_combined.shape[0]} registros")
        
        # 4. Ingeniería de características en dataset combinado
        df_engineered = feature_engineering(df_combined)
        
        # 5. Separar nuevamente train y test
        df_train_engineered = df_engineered[df_engineered['dataset'] == 'train'].copy()
        df_test_engineered = df_engineered[df_engineered['dataset'] == 'test'].copy()
        
        # Eliminar columna auxiliar
        df_train_engineered = df_train_engineered.drop(['dataset'], axis=1)
        df_test_engineered = df_test_engineered.drop(['dataset'], axis=1)
        
        print(f"Datos de entrenamiento procesados: {df_train_engineered.shape}")
        print(f"Datos de prueba procesados: {df_test_engineered.shape}")
        
        # 6. Preprocesar datos de entrenamiento
        X, y, label_encoders, feature_columns = preprocess_data(df_train_engineered)
        
        # 7. Dividir datos de entrenamiento
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # 8. Entrenar modelo
        model = train_model(X_train, y_train)
        
        # 9. Evaluar modelo
        accuracy, auc_score = evaluate_model(model, X_test, y_test)
        
        # 10. Opcional: Generar predicciones para el conjunto de prueba completo
        print("\n" + "="*70)
        print("GENERANDO PREDICCIONES PARA CONJUNTO DE PRUEBA")
        print("="*70)
        
        # Preprocesar datos de prueba usando los mismos encoders
        # Nota: df_test_engineered no tiene columna 'Survived'
        X_test_full = df_test_engineered.copy()
        
        # Seleccionar características para el modelo
        feature_columns_input = [
            'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'Title', 'FamilySize', 'IsAlone', 'AgeGroup', 'FareGroup', 'HasCabin', 'Deck'
        ]
        
        X_test_full = X_test_full[feature_columns_input]
        
        # Codificar variables categóricas usando los encoders ya entrenados
        categorical_columns = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'Deck']
        
        for col in categorical_columns:
            if col in X_test_full.columns and col in label_encoders:
                # Manejar valores no vistos durante el entrenamiento
                try:
                    X_test_full[col] = label_encoders[col].transform(X_test_full[col])
                except ValueError:
                    # Si hay valores no vistos, usar el más común
                    most_common = label_encoders[col].classes_[0]
                    X_test_full[col] = X_test_full[col].fillna(most_common).astype(str)
                    mask = ~X_test_full[col].isin(label_encoders[col].classes_)
                    X_test_full.loc[mask, col] = most_common
                    X_test_full[col] = label_encoders[col].transform(X_test_full[col])
        
        # Asegurar que las columnas coincidan
        missing_cols = set(feature_columns) - set(X_test_full.columns)
        for col in missing_cols:
            X_test_full[col] = 0
        
        X_test_full = X_test_full[feature_columns]
        
        # Generar predicciones
        test_predictions = model.predict(X_test_full)
        test_probabilities = model.predict_proba(X_test_full)[:, 1]
        
        # Crear archivo de submisión
        submission = pd.DataFrame({
            'PassengerId': df_test['PassengerId'],
            'Survived': test_predictions
        })
        
        submission_path = 'titanic_predictions.csv'
        submission.to_csv(submission_path, index=False)
        print(f"Predicciones guardadas en: {submission_path}")
        print(f"Predicciones de supervivencia: {test_predictions.sum()}/{len(test_predictions)} ({test_predictions.mean():.1%})")
        
        # 11. Guardar modelo, encoders y características
        save_model_and_encoders(model, label_encoders, feature_columns, 
                               model_path, encoders_path, features_path)
        
        print(f"\n{'='*70}")
        print("PIPELINE DE PREDICCIÓN DE SUPERVIVENCIA DEL TITANIC COMPLETADO")
        print(f"Precisión final del modelo: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Modelo guardado en: {model_path}")
        print(f"Encoders guardados en: {encoders_path}")
        print(f"Características guardadas en: {features_path}")
        print(f"{'='*70}")
        
        # Consejos de interpretación
        print(f"\n⚓ Interpretación de resultados del Titanic:")
        print(f"• Precisión > 0.80: Excelente modelo histórico")
        print(f"• AUC > 0.85: Muy buena capacidad discriminativa")
        print(f"• El modelo identifica patrones de supervivencia")
        print(f"• Útil para análisis histórico y educativo")
        
        # Factores clave de supervivencia
        print(f"\n🚢 Factores históricos clave:")
        print(f"• Género: Las mujeres tuvieron mayor supervivencia")
        print(f"• Clase: Primera clase tuvo mayor supervivencia")
        print(f"• Edad: Los niños tuvieron prioridad")
        print(f"• Familia: El tamaño de familia influyó en la supervivencia")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        return

if __name__ == "__main__":
    main()