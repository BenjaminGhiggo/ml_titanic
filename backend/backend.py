import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Union

class TitanicSurvivalPredictor:
    """Clase para cargar el modelo y realizar predicciones de supervivencia del Titanic"""
    
    def __init__(self, model_path='model/modelo_titanic.pkl', encoders_path='model/label_encoders_titanic.pkl', 
                 features_path='model/feature_columns.pkl'):
        """
        Inicializa el predictor cargando el modelo, encoders y características
        
        Args:
            model_path (str): Ruta al archivo del modelo entrenado
            encoders_path (str): Ruta al archivo de los label encoders
            features_path (str): Ruta al archivo de las columnas de características
        """
        self.model_path = model_path
        self.encoders_path = encoders_path
        self.features_path = features_path
        self.model = None
        self.label_encoders = None
        self.feature_columns = None
        
        # Definir las columnas de entrada esperadas
        self.input_columns = [
            'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
        ]
        
        # Columnas categóricas que necesitan encoding
        self.categorical_columns = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'Deck']
        
        self.load_model()
    
    def load_model(self):
        """Carga el modelo entrenado, encoders y características"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"No se encontró el modelo en {self.model_path}")
            
            if not os.path.exists(self.encoders_path):
                raise FileNotFoundError(f"No se encontraron los encoders en {self.encoders_path}")
            
            if not os.path.exists(self.features_path):
                raise FileNotFoundError(f"No se encontraron las características en {self.features_path}")
            
            print("Cargando modelo del Titanic...")
            self.model = joblib.load(self.model_path)
            print("Modelo cargado exitosamente!")
            
            print("Cargando encoders...")
            self.label_encoders = joblib.load(self.encoders_path)
            print("Encoders cargados exitosamente!")
            
            print("Cargando características...")
            self.feature_columns = joblib.load(self.features_path)
            print("Características cargadas exitosamente!")
            
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            raise
    
    def validate_input_data(self, data: Dict) -> Dict:
        """
        Valida que los datos de entrada tengan el formato correcto
        
        Args:
            data (Dict): Diccionario con los datos de entrada
            
        Returns:
            Dict: Datos validados y procesados
        """
        # Verificar que todas las columnas requeridas estén presentes
        missing_columns = set(self.input_columns) - set(data.keys())
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas: {missing_columns}")
        
        # Crear una copia de los datos para procesamiento
        processed_data = data.copy()
        
        # Validar tipos de datos numéricos
        numeric_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        for col in numeric_columns:
            try:
                processed_data[col] = float(processed_data[col])
            except (ValueError, TypeError):
                raise ValueError(f"El valor de '{col}' debe ser numérico, recibido: {data[col]}")
        
        # Validar rangos de datos
        if processed_data['Pclass'] not in [1, 2, 3]:
            raise ValueError("La clase (Pclass) debe ser 1, 2 o 3")
        
        if processed_data['Age'] < 0 or processed_data['Age'] > 100:
            raise ValueError("La edad debe estar entre 0 y 100 años")
        
        if processed_data['SibSp'] < 0:
            raise ValueError("El número de hermanos/esposos a bordo no puede ser negativo")
        
        if processed_data['Parch'] < 0:
            raise ValueError("El número de padres/hijos a bordo no puede ser negativo")
        
        if processed_data['Fare'] < 0:
            raise ValueError("El precio del boleto no puede ser negativo")
        
        # Validar valores categóricos
        if processed_data['Sex'] not in ['male', 'female']:
            raise ValueError("El sexo debe ser 'male' o 'female'")
        
        if processed_data['Embarked'] not in ['S', 'C', 'Q']:
            raise ValueError("El puerto de embarque debe ser 'S', 'C' o 'Q'")
        
        return processed_data
    
    def feature_engineering(self, data: Dict) -> Dict:
        """
        Aplica la misma ingeniería de características que se usó en el entrenamiento
        
        Args:
            data (Dict): Datos validados
            
        Returns:
            Dict: Datos con características ingeniería aplicada
        """
        # Crear características derivadas
        processed_data = data.copy()
        
        # 1. Crear Title basado en información disponible (simplificado)
        # Para predicción, usaremos reglas simples basadas en sexo y edad
        if processed_data['Sex'] == 'male':
            if processed_data['Age'] < 16:
                processed_data['Title'] = 'Master'
            else:
                processed_data['Title'] = 'Mr'
        else:  # female
            if processed_data['Age'] < 16:
                processed_data['Title'] = 'Miss'
            else:
                processed_data['Title'] = 'Mrs'  # Simplificación
        
        # 2. FamilySize
        processed_data['FamilySize'] = processed_data['SibSp'] + processed_data['Parch'] + 1
        
        # 3. IsAlone
        processed_data['IsAlone'] = 1 if processed_data['FamilySize'] == 1 else 0
        
        # 4. AgeGroup
        age = processed_data['Age']
        if age <= 12:
            processed_data['AgeGroup'] = 'Child'
        elif age <= 18:
            processed_data['AgeGroup'] = 'Teen'
        elif age <= 35:
            processed_data['AgeGroup'] = 'Adult'
        elif age <= 60:
            processed_data['AgeGroup'] = 'Middle'
        else:
            processed_data['AgeGroup'] = 'Senior'
        
        # 5. FareGroup - crear grupos basados en cuartiles típicos del Titanic
        fare = processed_data['Fare']
        if fare <= 7.91:
            processed_data['FareGroup'] = 'Low'
        elif fare <= 14.45:
            processed_data['FareGroup'] = 'Medium'
        elif fare <= 31.0:
            processed_data['FareGroup'] = 'High'
        else:
            processed_data['FareGroup'] = 'VeryHigh'
        
        # 6. HasCabin - para predicción, asumimos que no tiene información de cabina
        processed_data['HasCabin'] = 0
        
        # 7. Deck - sin información de cabina, asignamos 'Unknown'
        processed_data['Deck'] = 'Unknown'
        
        return processed_data
    
    def preprocess_for_prediction(self, data: Dict) -> np.ndarray:
        """
        Preprocesa los datos para que estén en el formato correcto para el modelo
        
        Args:
            data (Dict): Datos con ingeniería de características aplicada
            
        Returns:
            np.ndarray: Array con los datos listos para predicción
        """
        # Crear DataFrame con los datos
        df = pd.DataFrame([data])
        
        # Codificar variables categóricas usando los encoders entrenados
        for col in self.categorical_columns:
            if col in df.columns and col in self.label_encoders:
                # Manejar valores no vistos durante el entrenamiento
                try:
                    df[col] = self.label_encoders[col].transform(df[col])
                except ValueError:
                    # Si el valor no se vio durante el entrenamiento, usar el más común
                    most_common = self.label_encoders[col].classes_[0]
                    df[col] = df[col].fillna(most_common).astype(str)
                    df.loc[~df[col].isin(self.label_encoders[col].classes_), col] = most_common
                    df[col] = self.label_encoders[col].transform(df[col])
        
        # Asegurar que las columnas estén en el orden correcto y todas presentes
        for col in self.feature_columns:
            if col not in df.columns:
                # Si falta alguna columna, llenar con 0 o valor por defecto
                df[col] = 0
        
        df = df[self.feature_columns]
        
        return df.values
    
    def predict(self, data: Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
        """
        Realiza predicciones para uno o más registros
        
        Args:
            data: Diccionario con datos de un registro o lista de diccionarios
            
        Returns:
            Diccionario con la predicción y probabilidad, o lista de diccionarios
        """
        if self.model is None:
            raise RuntimeError("El modelo no ha sido cargado correctamente")
        
        # Manejar entrada individual vs múltiple
        is_single_prediction = isinstance(data, dict)
        if is_single_prediction:
            data = [data]
        
        results = []
        
        for record in data:
            try:
                # Validar datos
                validated_data = self.validate_input_data(record)
                
                # Aplicar ingeniería de características
                engineered_data = self.feature_engineering(validated_data)
                
                # Preprocesar para predicción
                processed_data = self.preprocess_for_prediction(engineered_data)
                
                # Realizar predicción
                prediction = self.model.predict(processed_data)[0]
                prediction_proba = self.model.predict_proba(processed_data)[0]
                
                # Preparar resultado
                result = {
                    'prediction': int(prediction),
                    'prediction_label': 'Habría Sobrevivido' if prediction == 1 else 'No Habría Sobrevivido',
                    'survival_probability': float(prediction_proba[1]),
                    'death_probability': float(prediction_proba[0]),
                    'confidence': float(max(prediction_proba)),
                    'survival_chance': self._get_survival_category(float(prediction_proba[1]))
                }
                
                results.append(result)
                
            except Exception as e:
                # En caso de error, agregar resultado con error
                results.append({
                    'error': str(e),
                    'prediction': None,
                    'prediction_label': 'Error en predicción'
                })
        
        # Retornar resultado individual o lista según la entrada
        return results[0] if is_single_prediction else results
    
    def predict_with_explanation(self, data: Dict) -> Dict:
        """
        Realiza predicción con explicación adicional
        
        Args:
            data (Dict): Datos del registro
            
        Returns:
            Dict: Resultado con predicción y explicación
        """
        result = self.predict(data)
        
        if 'error' in result:
            return result
        
        # Añadir explicación basada en factores históricos conocidos
        explanation = self._generate_explanation(data, result)
        result['explanation'] = explanation
        result['historical_factors'] = self._analyze_historical_factors(data)
        
        return result
    
    def _get_survival_category(self, survival_prob: float) -> str:
        """Determina la categoría de supervivencia basada en la probabilidad"""
        if survival_prob < 0.25:
            return "Muy Baja"
        elif survival_prob < 0.5:
            return "Baja"
        elif survival_prob < 0.75:
            return "Alta"
        else:
            return "Muy Alta"
    
    def _generate_explanation(self, data: Dict, prediction_result: Dict) -> str:
        """Genera una explicación histórica de la predicción"""
        factors = []
        
        # Analizar factores históricos conocidos del Titanic
        if data['Sex'] == 'female':
            factors.append("mujer (política 'mujeres y niños primero')")
        else:
            factors.append("hombre (menor prioridad en evacuación)")
        
        if data['Pclass'] == 1:
            factors.append("primera clase (acceso privilegiado a botes salvavidas)")
        elif data['Pclass'] == 2:
            factors.append("segunda clase (acceso moderado a botes salvavidas)")
        else:
            factors.append("tercera clase (acceso limitado a botes salvavidas)")
        
        if data['Age'] < 16:
            factors.append("menor de edad (prioridad en evacuación)")
        elif data['Age'] > 60:
            factors.append("adulto mayor (dificultades físicas)")
        
        family_size = data['SibSp'] + data['Parch'] + 1
        if family_size == 1:
            factors.append("viajaba solo")
        elif family_size <= 4:
            factors.append("familia pequeña (ventaja para moverse)")
        else:
            factors.append("familia grande (dificultad para coordinarse)")
        
        if data['Fare'] > 50:
            factors.append("boleto costoso (posible mejor ubicación)")
        elif data['Fare'] < 10:
            factors.append("boleto barato (posible ubicación desfavorable)")
        
        if prediction_result['prediction'] == 1:
            return f"HABRÍA SOBREVIVIDO (probabilidad: {prediction_result['survival_probability']:.1%}). Factores favorables: {', '.join(factors)}."
        else:
            return f"NO HABRÍA SOBREVIVIDO (probabilidad: {prediction_result['death_probability']:.1%}). Factores desfavorables: {', '.join(factors)}."
    
    def _analyze_historical_factors(self, data: Dict) -> Dict:
        """Analiza factores históricos específicos del Titanic"""
        analysis = {
            'class_factor': self._analyze_class_factor(data['Pclass']),
            'gender_factor': self._analyze_gender_factor(data['Sex']),
            'age_factor': self._analyze_age_factor(data['Age']),
            'family_factor': self._analyze_family_factor(data['SibSp'] + data['Parch']),
            'fare_factor': self._analyze_fare_factor(data['Fare'])
        }
        
        return analysis
    
    def _analyze_class_factor(self, pclass: int) -> Dict:
        """Analiza el factor de clase social"""
        class_analysis = {
            1: {"description": "Primera Clase", "survival_rate": "62%", "advantages": ["Mejor ubicación", "Acceso prioritario a botes"]},
            2: {"description": "Segunda Clase", "survival_rate": "47%", "advantages": ["Ubicación moderada", "Acceso limitado"]},
            3: {"description": "Tercera Clase", "survival_rate": "24%", "advantages": ["Ubicación inferior", "Acceso restringido"]}
        }
        return class_analysis[pclass]
    
    def _analyze_gender_factor(self, sex: str) -> Dict:
        """Analiza el factor de género"""
        if sex == 'female':
            return {
                "description": "Mujer",
                "survival_rate": "74%",
                "advantages": ["Política 'mujeres y niños primero'", "Prioridad en evacuación"]
            }
        else:
            return {
                "description": "Hombre",
                "survival_rate": "19%",
                "advantages": ["Menor prioridad", "Esperaban que las mujeres abordaran primero"]
            }
    
    def _analyze_age_factor(self, age: float) -> Dict:
        """Analiza el factor de edad"""
        if age < 16:
            return {
                "description": "Menor de edad",
                "survival_rate": "54%",
                "advantages": ["Prioridad junto con mujeres", "Protección especial"]
            }
        elif age > 60:
            return {
                "description": "Adulto mayor",
                "survival_rate": "23%",
                "advantages": ["Posibles dificultades físicas", "Menor agilidad"]
            }
        else:
            return {
                "description": "Adulto",
                "survival_rate": "38%",
                "advantages": ["Condición física estándar"]
            }
    
    def _analyze_family_factor(self, family_members: int) -> Dict:
        """Analiza el factor familiar"""
        if family_members == 0:
            return {
                "description": "Viajaba solo",
                "survival_rate": "30%",
                "advantages": ["Mayor libertad de movimiento", "No coordinación familiar requerida"]
            }
        elif family_members <= 3:
            return {
                "description": "Familia pequeña",
                "survival_rate": "58%",
                "advantages": ["Apoyo mutuo", "Fácil coordinación"]
            }
        else:
            return {
                "description": "Familia grande",
                "survival_rate": "16%",
                "advantages": ["Dificultad para coordinarse", "Separación posible"]
            }
    
    def _analyze_fare_factor(self, fare: float) -> Dict:
        """Analiza el factor del precio del boleto"""
        if fare > 50:
            return {
                "description": "Boleto costoso",
                "survival_rate": "65%",
                "advantages": ["Mejor ubicación en el barco", "Posible cabina superior"]
            }
        elif fare < 10:
            return {
                "description": "Boleto económico",
                "survival_rate": "25%",
                "advantages": ["Ubicación inferior", "Mayor distancia a botes salvavidas"]
            }
        else:
            return {
                "description": "Boleto moderado",
                "survival_rate": "45%",
                "advantages": ["Ubicación intermedia"]
            }

def main():
    """Función para probar el backend"""
    try:
        # Inicializar predictor
        predictor = TitanicSurvivalPredictor()
        
        # Datos de prueba - basados en pasajeros reales
        test_data = {
            'Pclass': 1,
            'Sex': 'female',
            'Age': 38,
            'SibSp': 1,
            'Parch': 0,
            'Fare': 71.28,
            'Embarked': 'C'
        }
        
        print("Datos de prueba (basados en pasajero real):")
        for key, value in test_data.items():
            print(f"  {key}: {value}")
        
        # Realizar predicción
        result = predictor.predict_with_explanation(test_data)
        
        print("\nResultado de la predicción:")
        for key, value in result.items():
            if key == 'historical_factors':
                print(f"  {key}:")
                for factor, details in value.items():
                    print(f"    {factor}: {details}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error en la prueba: {str(e)}")

if __name__ == "__main__":
    main()