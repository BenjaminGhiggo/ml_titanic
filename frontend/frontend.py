import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Agregar el directorio backend al path para importar
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

try:
    from backend import TitanicSurvivalPredictor
except ImportError as e:
    st.error(f"Error al importar el backend: {e}")
    st.stop()

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Supervivencia del Titanic",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para inicializar el predictor
@st.cache_resource
def load_predictor():
    """Carga el predictor de manera cached para mejor rendimiento"""
    try:
        predictor = TitanicSurvivalPredictor()
        return predictor, None
    except Exception as e:
        return None, str(e)

# Función para crear gráficos de probabilidad de supervivencia
def create_survival_probability_chart(result):
    """Crea un gráfico de barras con las probabilidades de supervivencia"""
    probabilities = [result['death_probability'], result['survival_probability']]
    labels = ['No Supervivencia', 'Supervivencia']
    colors = ['#8B0000', '#228B22']  # Rojo oscuro para muerte, verde para supervivencia
    
    fig = go.Figure(data=[
        go.Bar(x=labels, y=probabilities, marker_color=colors, 
               text=[f"{p:.1%}" for p in probabilities], textposition='auto')
    ])
    
    fig.update_layout(
        title="Probabilidades de Supervivencia en el Titanic",
        yaxis_title="Probabilidad",
        xaxis_title="Resultado",
        showlegend=False,
        height=400,
        yaxis=dict(tickformat='.0%')
    )
    
    return fig

# Función para crear gauge de supervivencia
def create_survival_gauge(survival_prob):
    """Crea un gauge que muestra la probabilidad de supervivencia"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = survival_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidad de Supervivencia (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 25], 'color': "darkred"},
                {'range': [25, 50], 'color': "red"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "blue", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# Función para mostrar información histórica del pasajero
def display_passenger_info(data):
    """Muestra información del pasajero simulado"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        class_name = {1: "Primera", 2: "Segunda", 3: "Tercera"}[data['Pclass']]
        st.metric("Clase", f"{class_name} ({data['Pclass']})")
    
    with col2:
        gender_es = "Mujer" if data['Sex'] == 'female' else "Hombre"
        st.metric("Género", gender_es)
    
    with col3:
        st.metric("Edad", f"{data['Age']} años")
    
    with col4:
        st.metric("Precio Boleto", f"£{data['Fare']:.2f}")

def create_historical_comparison_chart():
    """Crea un gráfico de comparación de tasas de supervivencia históricas"""
    # Datos históricos reales del Titanic
    historical_data = {
        'Categoría': ['Hombres', 'Mujeres', 'Niños', '1ra Clase', '2da Clase', '3ra Clase'],
        'Supervivencia': [0.19, 0.74, 0.52, 0.62, 0.47, 0.24]
    }
    
    fig = px.bar(
        x=historical_data['Categoría'], 
        y=historical_data['Supervivencia'],
        title="Tasas de Supervivencia Históricas del Titanic",
        labels={'x': 'Categoría', 'y': 'Tasa de Supervivencia'},
        color=historical_data['Supervivencia'],
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(height=400, yaxis=dict(tickformat='.0%'))
    fig.update_traces(texttemplate='%{y:.0%}', textposition='outside')
    
    return fig

def main():
    """Función principal de la aplicación Streamlit"""
    
    # Título principal con tema náutico
    st.title("⚓ Predictor de Supervivencia del Titanic")
    st.markdown("*Sistema de Machine Learning basado en datos históricos del RMS Titanic (1912)*")
    st.markdown("---")
    
    # Cargar el predictor
    predictor, error = load_predictor()
    
    if error:
        st.error(f"No se pudo cargar el modelo: {error}")
        st.info("Asegúrate de que el modelo haya sido entrenado ejecutando el script modelo.py")
        st.stop()
    
    # Sidebar con información histórica
    with st.sidebar:
        st.header("🚢 Información Histórica")
        st.write("""
        El RMS Titanic se hundió el 15 de abril de 1912 durante su viaje inaugural. 
        Este sistema usa Machine Learning para predecir supervivencia basándose en 
        datos reales de los pasajeros.
        """)
        
        st.header("📊 Estadísticas Reales")
        st.write("""
        - **Pasajeros a bordo**: 2,224
        - **Supervivientes**: 710 (32%)
        - **Víctimas**: 1,514 (68%)
        - **Botes disponibles**: 20
        - **Capacidad botes**: 1,178 personas
        """)
        
        st.header("⚖️ Factores Clave")
        st.write("""
        - 🚺 **Género**: "Mujeres y niños primero"
        - 🎩 **Clase social**: Acceso diferenciado
        - 👶 **Edad**: Prioridad para menores
        - 👨‍👩‍👧‍👦 **Familia**: Tamaño influyó en supervivencia
        - 💰 **Precio boleto**: Indicador de ubicación
        """)
    
    # Crear dos columnas principales
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("👤 Datos del Pasajero")
        
        # Formulario de entrada histórica
        with st.form("titanic_form"):
            # Información básica
            st.subheader("Información Personal")
            col_basic1, col_basic2 = st.columns(2)
            
            with col_basic1:
                sex = st.selectbox(
                    "Género",
                    options=['male', 'female'],
                    format_func=lambda x: "Hombre" if x == 'male' else "Mujer",
                    index=1
                )
                
                age = st.slider(
                    "Edad",
                    min_value=0,
                    max_value=80,
                    value=30,
                    step=1,
                    help="Edad del pasajero en años"
                )
            
            with col_basic2:
                pclass = st.selectbox(
                    "Clase del Boleto",
                    options=[1, 2, 3],
                    format_func=lambda x: f"{x}° Clase",
                    index=0,
                    help="Clase social y ubicación en el barco"
                )
            
            # Información familiar
            st.subheader("Información Familiar")
            col_family1, col_family2 = st.columns(2)
            
            with col_family1:
                sibsp = st.number_input(
                    "Hermanos/Esposos a bordo",
                    min_value=0,
                    max_value=8,
                    value=0,
                    step=1,
                    help="Número de hermanos o esposos/esposas a bordo"
                )
            
            with col_family2:
                parch = st.number_input(
                    "Padres/Hijos a bordo",
                    min_value=0,
                    max_value=6,
                    value=0,
                    step=1,
                    help="Número de padres o hijos a bordo"
                )
            
            # Información del viaje
            st.subheader("Información del Viaje")
            col_travel1, col_travel2 = st.columns(2)
            
            with col_travel1:
                fare = st.number_input(
                    "Precio del Boleto (£)",
                    min_value=0.0,
                    max_value=512.0,
                    value=32.0,
                    step=0.25,
                    help="Precio pagado por el boleto en libras esterlinas"
                )
            
            with col_travel2:
                embarked = st.selectbox(
                    "Puerto de Embarque",
                    options=['S', 'C', 'Q'],
                    format_func=lambda x: {
                        'S': 'Southampton',
                        'C': 'Cherbourg', 
                        'Q': 'Queenstown'
                    }[x],
                    index=0,
                    help="Puerto donde abordó el Titanic"
                )
            
            # Botón de predicción
            submitted = st.form_submit_button("🔮 Predecir Supervivencia", use_container_width=True)
    
    with col2:
        st.header("📊 Análisis de Supervivencia")
        
        if submitted:
            # Preparar datos para predicción
            input_data = {
                'Pclass': pclass,
                'Sex': sex,
                'Age': age,
                'SibSp': sibsp,
                'Parch': parch,
                'Fare': fare,
                'Embarked': embarked
            }
            
            # Mostrar información del pasajero
            display_passenger_info(input_data)
            
            # Mostrar datos de entrada en expander
            with st.expander("Ver datos completos del pasajero"):
                passenger_df = pd.DataFrame([{
                    'Clase': f"{pclass}° Clase",
                    'Género': "Mujer" if sex == 'female' else "Hombre",
                    'Edad': f"{age} años",
                    'Hermanos/Esposos': sibsp,
                    'Padres/Hijos': parch,
                    'Precio Boleto': f"£{fare:.2f}",
                    'Puerto Embarque': {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}[embarked]
                }])
                st.dataframe(passenger_df, use_container_width=True)
            
            # Realizar predicción
            with st.spinner("Analizando probabilidad de supervivencia..."):
                try:
                    result = predictor.predict_with_explanation(input_data)
                    
                    if 'error' in result:
                        st.error(f"Error en la predicción: {result['error']}")
                    else:
                        # Mostrar resultado principal
                        survival_chance = result['survival_chance']
                        
                        if result['prediction'] == 1:
                            if survival_chance == "Muy Alta":
                                st.success(f"⚓ **{result['prediction_label']}** - Probabilidad: {survival_chance}")
                                st.balloons()
                            else:
                                st.success(f"✅ **{result['prediction_label']}** - Probabilidad: {survival_chance}")
                        else:
                            if survival_chance == "Muy Baja":
                                st.error(f"💀 **{result['prediction_label']}** - Probabilidad: {survival_chance}")
                            else:
                                st.warning(f"⚠️ **{result['prediction_label']}** - Probabilidad: {survival_chance}")
                        
                        # Mostrar métricas de predicción
                        col_metrics1, col_metrics2 = st.columns(2)
                        
                        with col_metrics1:
                            st.metric(
                                "Probabilidad de Supervivencia",
                                f"{result['survival_probability']:.1%}",
                                delta=f"{result['survival_probability'] - 0.32:.1%}",
                                help="Comparado con la tasa histórica general (32%)"
                            )
                        
                        with col_metrics2:
                            st.metric(
                                "Confianza del Modelo",
                                f"{result['confidence']:.1%}",
                                delta=None
                            )
                        
                        # Gráfico de probabilidades
                        fig_prob = create_survival_probability_chart(result)
                        st.plotly_chart(fig_prob, use_container_width=True)
                        
                        # Gauge de supervivencia
                        fig_survival = create_survival_gauge(result['survival_probability'])
                        st.plotly_chart(fig_survival, use_container_width=True)
                        
                        # Explicación histórica
                        if 'explanation' in result:
                            st.subheader("🔍 Análisis Histórico")
                            st.info(result['explanation'])
                        
                        # Factores históricos detallados
                        if 'historical_factors' in result and result['historical_factors']:
                            st.subheader("⚖️ Análisis de Factores Históricos")
                            
                            # Crear tabs para diferentes factores
                            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                                "👑 Clase", "👤 Género", "🎂 Edad", "👨‍👩‍👧‍👦 Familia", "💰 Boleto"
                            ])
                            
                            factors = result['historical_factors']
                            
                            with tab1:
                                if 'class_factor' in factors:
                                    cf = factors['class_factor']
                                    st.write(f"**{cf['description']}**")
                                    st.write(f"Tasa histórica de supervivencia: **{cf['survival_rate']}**")
                                    st.write("Ventajas/Desventajas:")
                                    for adv in cf['advantages']:
                                        st.write(f"• {adv}")
                            
                            with tab2:
                                if 'gender_factor' in factors:
                                    gf = factors['gender_factor']
                                    st.write(f"**{gf['description']}**")
                                    st.write(f"Tasa histórica de supervivencia: **{gf['survival_rate']}**")
                                    st.write("Factores:")
                                    for adv in gf['advantages']:
                                        st.write(f"• {adv}")
                            
                            with tab3:
                                if 'age_factor' in factors:
                                    af = factors['age_factor']
                                    st.write(f"**{af['description']}**")
                                    st.write(f"Tasa histórica de supervivencia: **{af['survival_rate']}**")
                                    st.write("Factores:")
                                    for adv in af['advantages']:
                                        st.write(f"• {adv}")
                            
                            with tab4:
                                if 'family_factor' in factors:
                                    ff = factors['family_factor']
                                    st.write(f"**{ff['description']}**")
                                    st.write(f"Tasa histórica de supervivencia: **{ff['survival_rate']}**")
                                    st.write("Factores:")
                                    for adv in ff['advantages']:
                                        st.write(f"• {adv}")
                            
                            with tab5:
                                if 'fare_factor' in factors:
                                    faf = factors['fare_factor']
                                    st.write(f"**{faf['description']}**")
                                    st.write(f"Tasa histórica de supervivencia: **{faf['survival_rate']}**")
                                    st.write("Factores:")
                                    for adv in faf['advantages']:
                                        st.write(f"• {adv}")
                        
                        # Contexto histórico adicional
                        st.subheader("📚 Contexto Histórico")
                        family_size = sibsp + parch + 1
                        
                        embarked_names = {
                            'S': 'Southampton (Inglaterra)', 
                            'C': 'Cherbourg (Francia)', 
                            'Q': 'Queenstown (Irlanda)'
                        }
                        
                        context_info = f"""
                        **Perfil del Pasajero Simulado:**
                        - Viajaba {'solo' if family_size == 1 else f'con {family_size-1} familiares'}
                        - Pagó £{fare:.2f} por su boleto ({'caro' if fare > 50 else 'barato' if fare < 15 else 'moderado'})
                        - Embarcó en {embarked_names[embarked]}
                        - Su clase social le daba acceso a {'cubiertas superiores' if pclass == 1 else 'cubiertas intermedias' if pclass == 2 else 'cubiertas inferiores'}
                        """
                        st.write(context_info)
                
                except Exception as e:
                    st.error(f"Error inesperado: {str(e)}")
        
        else:
            st.info("👆 Complete la información del pasajero y haga clic en 'Predecir Supervivencia' para ver los resultados.")
            
            # Mostrar gráfico de comparación histórica
            st.subheader("📊 Tasas Históricas de Supervivencia")
            fig_historical = create_historical_comparison_chart()
            st.plotly_chart(fig_historical, use_container_width=True)
    
    # Información adicional en la parte inferior
    st.markdown("---")
    
    with st.expander("🎓 Información Educativa sobre el Titanic"):
        col_edu1, col_edu2, col_edu3 = st.columns(3)
        
        with col_edu1:
            st.write("""
            **🚢 Datos del Barco:**
            - Longitud: 269 metros
            - Tripulación: 885 personas
            - Velocidad máxima: 24 nudos
            - Construcción: 1909-1912
            - Costo: £1.5 millones
            """)
        
        with col_edu2:
            st.write("""
            **❄️ El Desastre:**
            - Fecha: 14-15 abril 1912
            - Hora impacto: 23:40
            - Tiempo hundimiento: 2h 40min
            - Causa: Choque con iceberg
            - Temperatura agua: -2°C
            """)
        
        with col_edu3:
            st.write("""
            **🛟 Botes Salvavidas:**
            - Botes disponibles: 20
            - Capacidad total: 1,178
            - Pasajeros evacuados: 710
            - Último bote: 02:05
            - Rescate: RMS Carpathia
            """)
    
    # Sección de Machine Learning
    st.subheader("🤖 Sobre el Modelo de Machine Learning")
    col_ml1, col_ml2, col_ml3 = st.columns(3)
    
    with col_ml1:
        st.metric("Algoritmo", "Random Forest", help="Conjunto de árboles de decisión")
    
    with col_ml2:
        st.metric("Precisión", "~85%", help="Porcentaje de predicciones correctas")
    
    with col_ml3:
        st.metric("Características", "14", help="Variables utilizadas para la predicción")
    
    # Footer histórico
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
        <p>⚓ Predictor de Supervivencia del Titanic - Análisis Histórico con Machine Learning</p>
        <p>Basado en datos reales de los 891 pasajeros del RMS Titanic</p>
        <p><em>"Even God himself couldn't sink this ship" - En memoria de las 1,514 víctimas</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()