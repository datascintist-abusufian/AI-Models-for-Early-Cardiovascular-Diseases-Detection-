import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def initialize_session_state():
    """Initialize session state variables for prediction page"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model_performance' not in st.session_state:
        st.session_state.model_performance = None
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None

initialize_session_state()

# ============================================================================
# MODEL LOADING WITH ERROR HANDLING
# ============================================================================
@st.cache_resource
def load_model():
    """Load the pre-trained model with error handling"""
    try:
        model_filename = 'model.pkl'
        
        # Check if model file exists
        if not os.path.exists(model_filename):
            st.error(f"❌ Model file '{model_filename}' not found!")
            st.info("💡 Please ensure the model file exists in the current directory.")
            return None
        
        # Try different loading methods
        try:
            with open(model_filename, 'rb') as file:
                model = pickle.load(file)
                st.session_state.model_loaded = True
                return model
        except pickle.UnpicklingError:
            st.error("❌ Error loading model file. The file might be corrupted.")
            return None
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"❌ Unexpected error loading model: {str(e)}")
        return None

@st.cache_resource
def load_feature_importance():
    """Load or calculate feature importance"""
    try:
        # Try to load pre-calculated feature importance
        if os.path.exists('feature_importance.json'):
            with open('feature_importance.json', 'r') as f:
                return json.load(f)
        else:
            # Return default feature importance (if available)
            return None
    except:
        return None

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def make_prediction(model, user_input):
    """Make prediction with confidence scores"""
    try:
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)
        
        # Get confidence
        if prediction[0] == 1:
            confidence = prediction_proba[0][1]
            result = 'Positive'
            risk_level = 'High' if confidence > 0.7 else 'Moderate'
        else:
            confidence = prediction_proba[0][0]
            result = 'Negative'
            risk_level = 'Low' if confidence > 0.7 else 'Moderate'
        
        return {
            'prediction': prediction[0],
            'result': result,
            'confidence': float(confidence),
            'risk_level': risk_level,
            'probabilities': {
                'negative': float(prediction_proba[0][0]),
                'positive': float(prediction_proba[0][1])
            }
        }
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None

def save_prediction_history(user_input, prediction_result):
    """Save prediction to history"""
    history_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'age': user_input['age'][0],
        'sex': 'Male' if user_input['sex'][0] == 1 else 'Female',
        'prediction': prediction_result['result'],
        'confidence': f"{prediction_result['confidence']*100:.1f}%",
        'risk_level': prediction_result['risk_level']
    }
    st.session_state.prediction_history.append(history_entry)

# ============================================================================
# UI COMPONENTS
# ============================================================================
def display_logo():
    """Display logo and header"""
    logo = "logo Heart.jpeg"
    if os.path.exists(logo):
        st.image(logo, use_column_width=True, caption="Cardiovascular")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #ff4b4b, #ff6b6b); border-radius: 10px;">
            <h1 style="color: white;">❤️ AI Models for Early Cardiovascular Diseases Prediction</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.title('AI-Models-for-Early-Cardiovascular-Diseases-Prediction')
    st.write("""### We need patient data to predict on...Author-Md Abu Sufian""")

def create_input_section():
    """Create the input section with organized layout"""
    st.subheader("📋 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider('Age', 18, 100, 50, help='Patient age in years')
        sex = st.selectbox('Sex', ['Male', 'Female'], help='Patient gender')
        cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        trestbps = st.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120, help='Resting blood pressure in mm Hg')
        chol = st.slider('Cholesterol (mg/dl)', 100, 600, 250, help='Serum cholesterol in mg/dl')
        
    with col2:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])
        restecg = st.selectbox('Resting ECG Results', ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'])
        thalach = st.slider('Max Heart Rate Achieved', 70, 220, 150, help='Maximum heart rate achieved')
        exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        oldpeak = st.slider('ST Depression', 0.0, 6.2, 1.0, step=0.1, help='ST depression induced by exercise relative to rest')
    
    st.subheader("📊 Additional Clinical Measurements")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        slope = st.selectbox('ST Segment Slope', ['Upsloping', 'Flat', 'Downsloping'])
    with col4:
        ca = st.slider('Major Vessels', 0, 4, 1, help='Number of major vessels colored by fluoroscopy')
    with col5:
        thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
    
    # Convert inputs to numeric values
    sex_num = 1 if sex == 'Male' else 0
    cp_num = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
    fbs_num = ['False', 'True'].index(fbs)
    restecg_num = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'].index(restecg)
    exang_num = ['No', 'Yes'].index(exang)
    slope_num = ['Upsloping', 'Flat', 'Downsloping'].index(slope)
    thal_num = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal)
    
    return {
        'age': age,
        'sex_num': sex_num,
        'cp_num': cp_num,
        'trestbps': trestbps,
        'chol': chol,
        'fbs_num': fbs_num,
        'restecg_num': restecg_num,
        'thalach': thalach,
        'exang_num': exang_num,
        'oldpeak': oldpeak,
        'slope_num': slope_num,
        'ca': ca,
        'thal_num': thal_num
    }

def display_prediction_result(prediction_result):
    """Display prediction results with visual enhancements"""
    if prediction_result is None:
        return
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Color coding based on prediction
    if prediction_result['result'] == 'Positive':
        bg_color = '#ff4b4b'
        text_color = 'white'
        icon = '⚠️'
    else:
        bg_color = '#28a745'
        text_color = 'white'
        icon = '✅'
    
    # Main result card
    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center; color: {text_color};">
        <h1>{icon} Prediction: {prediction_result['result']}</h1>
        <h3>Risk Level: {prediction_result['risk_level']}</h3>
        <h4>Confidence: {prediction_result['confidence']*100:.1f}%</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge chart for confidence
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction_result['confidence']*100,
            title={'text': "Confidence Score"},
            delta={'reference': 70, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#ff4b4b" if prediction_result['result'] == 'Positive' else "#28a745"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcccc"},
                    {'range': [50, 75], 'color': "#ffffcc"},
                    {'range': [75, 100], 'color': "#ccffcc"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Probability distribution
        prob_df = pd.DataFrame({
            'Class': ['Negative', 'Positive'],
            'Probability': [
                prediction_result['probabilities']['negative'],
                prediction_result['probabilities']['positive']
            ]
        })
        
        fig_probs = px.bar(
            prob_df,
            x='Class',
            y='Probability',
            color='Class',
            color_discrete_map={'Negative': '#28a745', 'Positive': '#ff4b4b'},
            title='Prediction Probabilities',
            range_y=[0, 1],
            text='Probability'
        )
        fig_probs.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_probs.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_probs, use_container_width=True)
    
    # Risk factors
    with st.expander("🔬 Risk Factor Analysis", expanded=False):
        risk_factors = {
            'Age': 'High risk for age > 60',
            'Chest Pain': 'Asymptomatic indicates higher risk',
            'Cholesterol': 'High cholesterol > 240 mg/dl',
            'Blood Pressure': 'High blood pressure > 140/90',
            'Max Heart Rate': 'Low max heart rate (< 100) increases risk',
            'Exercise Angina': 'Exercise-induced angina is a strong indicator'
        }
        
        for factor, description in risk_factors.items():
            st.info(f"**{factor}:** {description}")

def display_prediction_history():
    """Display prediction history"""
    if st.session_state.prediction_history:
        st.subheader("📜 Prediction History")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Style the dataframe
        def color_prediction(val):
            color = '#ff4b4b' if val == 'Positive' else '#28a745'
            return f'color: {color}; font-weight: bold'
        
        styled_df = history_df.style.applymap(
            color_prediction, 
            subset=['prediction']
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Download history
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediction History",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
        )

def display_model_info():
    """Display model information and performance metrics"""
    with st.expander("ℹ️ Model Information", expanded=False):
        st.markdown("""
        ### 🤖 Model Details
        
        **Model Type:** Machine Learning Classifier
        
        **Training Data:** 
        - Features: 13 clinical parameters
        - Samples: 303 patient records
        
        **Performance Metrics (on test set):**
        - Accuracy: ~85%
        - Precision: ~83%
        - Recall: ~87%
        - F1-Score: ~85%
        
        **Key Features:**
        1. Maximum Heart Rate (thalach)
        2. Chest Pain Type (cp)
        3. ST Depression (oldpeak)
        4. Number of Major Vessels (ca)
        5. Thalassemia (thal)
        """)
        
        # Feature importance visualization (if available)
        if st.session_state.feature_importance:
            st.subheader("📊 Feature Importance")
            import json
            if isinstance(st.session_state.feature_importance, dict):
                importance_df = pd.DataFrame.from_dict(
                    st.session_state.feature_importance, 
                    orient='index', 
                    columns=['Importance']
                ).sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y=importance_df.index,
                    orientation='h',
                    title='Feature Importance',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_importance, use_container_width=True)

def display_education_section():
    """Display educational content about heart disease"""
    with st.expander("📚 Learn More About Heart Disease", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🫀 Understanding Heart Disease
            
            Heart disease refers to various conditions that affect the heart's function.
            Common types include:
            
            - **Coronary Artery Disease**: Narrowing of the heart's blood vessels
            - **Heart Attack**: Blockage of blood flow to the heart
            - **Heart Failure**: Heart can't pump enough blood
            - **Arrhythmias**: Irregular heartbeats
            
            ### ⚠️ Risk Factors
            
            **Modifiable:**
            - Smoking
            - High blood pressure
            - High cholesterol
            - Diabetes
            - Obesity
            - Physical inactivity
            
            **Non-modifiable:**
            - Age
            - Gender
            - Family history
            - Race/ethnicity
            """)
        
        with col2:
            st.markdown("""
            ### 🩺 Prevention Tips
            
            **Lifestyle Changes:**
            1. 🥗 Eat a heart-healthy diet (Mediterranean diet)
            2. 🏃 Regular physical activity (150 min/week)
            3. 🚭 Quit smoking
            4. 🍷 Limit alcohol consumption
            5. 🧘 Manage stress
            6. 💤 Get adequate sleep (7-8 hours)
            
            ### 📊 Warning Signs
            
            **Seek immediate medical help if you experience:**
            - Chest pain or discomfort
            - Shortness of breath
            - Pain in arms, neck, jaw, or back
            - Nausea, lightheadedness
            - Cold sweats
            """)

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main prediction page function"""
    
    # Display header and logo
    display_logo()
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Load feature importance (if available)
    if st.session_state.feature_importance is None:
        st.session_state.feature_importance = load_feature_importance()
    
    # Create input section
    user_inputs = create_input_section()
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔬 Predict", use_container_width=True, type="primary")
    
    # Make prediction when button is pressed
    if predict_button:
        # Prepare input for model
        user_input = pd.DataFrame(data={
            'age': [user_inputs['age']],
            'sex': [user_inputs['sex_num']],
            'cp': [user_inputs['cp_num']],
            'trestbps': [user_inputs['trestbps']],
            'chol': [user_inputs['chol']],
            'fbs': [user_inputs['fbs_num']],
            'restecg': [user_inputs['restecg_num']],
            'thalach': [user_inputs['thalach']],
            'exang': [user_inputs['exang_num']],
            'oldpeak': [user_inputs['oldpeak']],
            'slope': [user_inputs['slope_num']],
            'ca': [user_inputs['ca']],
            'thal': [user_inputs['thal_num']]
        })
        
        with st.spinner('🧠 Making prediction...'):
            prediction_result = make_prediction(model, user_input)
            
        if prediction_result:
            # Save to history
            save_prediction_history(user_input, prediction_result)
            
            # Display results
            display_prediction_result(prediction_result)
    
    # Display prediction history
    display_prediction_history()
    
    # Display model info
    display_model_info()
    
    # Display educational content
    display_education_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🔬 AI Models for Early Cardiovascular Diseases Detection</p>
        <p>Version 2.0 | Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
