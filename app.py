import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #FF4B4B;
        text-align: center;
        padding: 1rem 0;
    }
    h2 {
        color: #0E1117;
        border-bottom: 2px solid #FF4B4B;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #6EC325;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    """Load saved models and scaler"""
    try:
        if os.path.exists('models/best_model_random_forest.pkl'):
            model = joblib.load('models/best_model_random_forest.pkl')
        elif os.path.exists('models/tuned_random_forest.pkl'):
            model = joblib.load('models/tuned_random_forest.pkl')
        else:
            st.error("⚠️ Model file not found! Please train the model first.")
            return None, None
        
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_data():
    """Load the heart disease dataset"""
    try:
        data = pd.read_csv('heart_disease_data.csv')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Initialize
model, scaler = load_models()
heart_data = load_data()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/cotton/128/000000/like--v1.png", width=100)
    st.title("🫀 Navigation")
    
    page = st.radio(
        "Choose a page:",
        ["🏠 Home", "📊 Data Analysis", "🤖 Model Performance", "🔮 Prediction", "📈 Visualizations", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    st.markdown(f"**User:** AyushMaurya13")
    st.markdown(f"**Version:** 2.0")

# Main content
if page == "🏠 Home":
    st.title("🫀 Heart Disease Prediction System")
    st.markdown("### Advanced Machine Learning Analysis for Cardiovascular Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Dataset</h3>
            <p><strong>303</strong> patients</p>
            <p><strong>13</strong> features</p>
            <p><strong>2</strong> classes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Models</h3>
            <p><strong>8</strong> algorithms</p>
            <p><strong>85%+</strong> accuracy</p>
            <p><strong>0.90+</strong> ROC-AUC</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>✨ Features</h3>
            <p>Cross-validation</p>
            <p>Hyperparameter tuning</p>
            <p>Real-time prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Project Overview")
    st.write("""
    This comprehensive heart disease prediction system uses advanced machine learning techniques to 
    predict the likelihood of heart disease based on patient health metrics. The system includes:
    
    - **Multi-Model Comparison**: 8 different ML algorithms evaluated
    - **Feature Engineering**: StandardScaler normalization
    - **Cross-Validation**: 5-fold stratified CV for robust evaluation
    - **Hyperparameter Tuning**: GridSearchCV optimization
    - **Interactive Predictions**: Real-time disease risk assessment
    """)
    
    if heart_data is not None:
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(heart_data.head(10), use_container_width=True)
        
        st.markdown("### 🔍 Feature Descriptions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - **age**: Age in years
            - **sex**: Sex (1 = male, 0 = female)
            - **cp**: Chest pain type (0-3)
            - **trestbps**: Resting blood pressure (mm Hg)
            - **chol**: Serum cholesterol (mg/dl)
            - **fbs**: Fasting blood sugar > 120 mg/dl
            - **restecg**: Resting ECG results (0-2)
            """)
        
        with col2:
            st.markdown("""
            - **thalach**: Maximum heart rate achieved
            - **exang**: Exercise induced angina (1 = yes)
            - **oldpeak**: ST depression induced by exercise
            - **slope**: Slope of peak exercise ST segment (0-2)
            - **ca**: Number of major vessels (0-4)
            - **thal**: Thalassemia (0-3)
            - **target**: Heart disease (1 = yes, 0 = no)
            """)

elif page == "📊 Data Analysis":
    st.title("📊 Data Analysis & Exploration")
    
    if heart_data is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Statistics", "🎯 Target Analysis", "🔗 Correlations", "📉 Distributions"])
        
        with tab1:
            st.markdown("### 📈 Statistical Summary")
            st.dataframe(heart_data.describe().T, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Samples", len(heart_data))
                st.metric("Missing Values", heart_data.isnull().sum().sum())
            with col2:
                st.metric("Features", len(heart_data.columns) - 1)
                st.metric("Duplicates", heart_data.duplicated().sum())
        
        with tab2:
            st.markdown("### 🎯 Target Variable Distribution")
            
            target_counts = heart_data['target'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=target_counts.values,
                    names=['No Disease', 'Disease'],
                    title='Target Distribution',
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=['No Disease', 'Disease'],
                    y=target_counts.values,
                    title='Target Count',
                    color=['No Disease', 'Disease'],
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("No Disease", target_counts[0], f"{target_counts[0]/len(heart_data)*100:.1f}%")
            with col2:
                st.metric("Disease", target_counts[1], f"{target_counts[1]/len(heart_data)*100:.1f}%")
            with col3:
                balance = target_counts[1] / target_counts[0]
                st.metric("Balance Ratio", f"{balance:.2f}", 
                         "✅ Balanced" if 0.8 <= balance <= 1.2 else "⚠️ Imbalanced")
        
        with tab3:
            st.markdown("### 🔗 Feature Correlations")
            
            correlation = heart_data.corr()
            
            fig = px.imshow(
                correlation,
                labels=dict(color="Correlation"),
                x=correlation.columns,
                y=correlation.columns,
                color_continuous_scale='RdBu_r',
                title='Correlation Heatmap'
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Top Correlations with Target")
            target_corr = correlation['target'].abs().sort_values(ascending=False)[1:11]
            
            fig = px.bar(
                x=target_corr.values,
                y=target_corr.index,
                orientation='h',
                title='Top 10 Features by Correlation with Target',
                labels={'x': 'Absolute Correlation', 'y': 'Feature'},
                color=target_corr.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("### 📉 Feature Distributions")
            
            numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            
            selected_feature = st.selectbox("Select Feature:", numerical_features)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    heart_data,
                    x=selected_feature,
                    color='target',
                    marginal='box',
                    title=f'{selected_feature.upper()} Distribution by Target',
                    labels={'target': 'Heart Disease'},
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    heart_data,
                    x='target',
                    y=selected_feature,
                    color='target',
                    title=f'{selected_feature.upper()} Box Plot',
                    labels={'target': 'Heart Disease'},
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance Analysis")
    
    # Load results if available
    if os.path.exists('models/model_comparison.csv'):
        results_df = pd.read_csv('models/model_comparison.csv')
        
        tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "📈 Metrics Details", "🎯 Best Model"])
        
        with tab1:
            st.markdown("### 📊 Model Performance Comparison")
            
            # Display results table
            st.dataframe(results_df.style.highlight_max(axis=0, subset=['Test Acc', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']), 
                        use_container_width=True)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    results_df.sort_values('Test Acc', ascending=False),
                    x='Model',
                    y='Test Acc',
                    title='Test Accuracy by Model',
                    color='Test Acc',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    results_df.sort_values('ROC-AUC', ascending=False),
                    x='Model',
                    y='ROC-AUC',
                    title='ROC-AUC Score by Model',
                    color='ROC-AUC',
                    color_continuous_scale='Plasma'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 📈 Detailed Metrics")
            
            # Train vs Test Accuracy
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Train Accuracy',
                x=results_df['Model'],
                y=results_df['Train Acc'],
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                name='Test Accuracy',
                x=results_df['Model'],
                y=results_df['Test Acc'],
                marker_color='darkblue'
            ))
            fig.update_layout(
                title='Train vs Test Accuracy',
                barmode='group',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Precision, Recall, F1-Score
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Precision', x=results_df['Model'], y=results_df['Precision']))
            fig.add_trace(go.Bar(name='Recall', x=results_df['Model'], y=results_df['Recall']))
            fig.add_trace(go.Bar(name='F1-Score', x=results_df['Model'], y=results_df['F1-Score']))
            fig.update_layout(
                title='Precision, Recall, F1-Score Comparison',
                barmode='group',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 🎯 Best Model Performance")
            
            best_model_name = results_df.iloc[0]['Model']
            best_row = results_df.iloc[0]
            
            st.success(f"🏆 **Best Model:** {best_model_name}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Test Accuracy", f"{best_row['Test Acc']:.4f}")
            with col2:
                st.metric("Precision", f"{best_row['Precision']:.4f}")
            with col3:
                st.metric("Recall", f"{best_row['Recall']:.4f}")
            with col4:
                st.metric("ROC-AUC", f"{best_row['ROC-AUC']:.4f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("F1-Score", f"{best_row['F1-Score']:.4f}")
                st.metric("Train Accuracy", f"{best_row['Train Acc']:.4f}")
            
            with col2:
                overfitting = best_row['Overfitting']
                st.metric("Overfitting", f"{overfitting:.4f}", 
                         "✅ Low" if overfitting < 0.05 else "⚠️ High")
                
                if os.path.exists('models/cross_validation.csv'):
                    cv_df = pd.read_csv('models/cross_validation.csv')
                    cv_row = cv_df[cv_df['Model'] == best_model_name].iloc[0]
                    st.metric("CV Score", f"{cv_row['Mean CV']:.4f} (±{cv_row['Std CV']:.4f})")
    
    else:
        st.warning("⚠️ Model comparison results not found. Please train the models first.")

elif page == "🔮 Prediction":
    st.title("🔮 Heart Disease Prediction")
    
    if model is not None and scaler is not None:
        st.markdown("### Enter Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (years)", min_value=29, max_value=77, value=50, step=1)
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"][x])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=94, max_value=200, value=120, step=1)
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=126, max_value=564, value=200, step=1)
        
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], 
                             format_func=lambda x: "No" if x == 0 else "Yes")
            restecg = st.selectbox("Resting ECG", options=[0, 1, 2], 
                                 format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
            thalach = st.number_input("Max Heart Rate", min_value=71, max_value=202, value=150, step=1)
            exang = st.selectbox("Exercise Induced Angina", options=[0, 1], 
                               format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col3:
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2], 
                               format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
            ca = st.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
            thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], 
                              format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])
        
        st.markdown("---")
        
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            # Prepare input
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                   thalach, exang, oldpeak, slope, ca, thal]])
            
            # Create DataFrame with feature names
            feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            input_df = pd.DataFrame(input_data, columns=feature_names)
            
            # Scale input
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(input_scaled)[0]
                prob_disease = probability[1]
            else:
                prob_disease = None
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            if prediction == 1:
                st.error("### ⚠️ HEART DISEASE DETECTED")
                st.markdown("**Recommendation:** Consult a cardiologist immediately for further evaluation.")
            else:
                st.success("### ✅ NO HEART DISEASE DETECTED")
                st.markdown("**Recommendation:** Maintain a healthy lifestyle and regular check-ups.")
            
            if prob_disease is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Disease Probability", f"{prob_disease*100:.2f}%")
                
                with col2:
                    st.metric("Healthy Probability", f"{(1-prob_disease)*100:.2f}%")
                
                with col3:
                    if prob_disease < 0.3:
                        risk = "🟢 LOW RISK"
                    elif prob_disease < 0.7:
                        risk = "🟡 MODERATE RISK"
                    else:
                        risk = "🔴 HIGH RISK"
                    st.metric("Risk Level", risk)
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob_disease * 100,
                    title={'text': "Disease Probability (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            # Show input summary
            with st.expander("📋 View Input Summary"):
                input_summary = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': input_data[0]
                })
                st.dataframe(input_summary, use_container_width=True)
    
    else:
        st.error("⚠️ Model not loaded. Please check if model files exist in the 'models' directory.")

elif page == "📈 Visualizations":
    st.title("📈 Advanced Visualizations")
    
    if heart_data is not None:
        viz_type = st.selectbox(
            "Select Visualization Type:",
            ["Age Analysis", "Gender Analysis", "Chest Pain Analysis", "Feature Relationships", "3D Analysis"]
        )
        
        if viz_type == "Age Analysis":
            st.markdown("### 👴 Age Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.violin(
                    heart_data,
                    x='target',
                    y='age',
                    color='target',
                    box=True,
                    title='Age Distribution by Target',
                    labels={'target': 'Heart Disease'},
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                heart_data['age_group'] = pd.cut(heart_data['age'], bins=[0, 40, 50, 60, 100], 
                                                  labels=['<40', '40-50', '50-60', '60+'])
                age_group_data = heart_data.groupby(['age_group', 'target']).size().reset_index(name='count')
                
                fig = px.bar(
                    age_group_data,
                    x='age_group',
                    y='count',
                    color='target',
                    title='Age Groups vs Heart Disease',
                    labels={'target': 'Heart Disease'},
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                heart_data.drop('age_group', axis=1, inplace=True)
        
        elif viz_type == "Gender Analysis":
            st.markdown("### 👥 Gender Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                gender_data = heart_data.groupby(['sex', 'target']).size().reset_index(name='count')
                
                fig = px.bar(
                    gender_data,
                    x='sex',
                    y='count',
                    color='target',
                    title='Gender vs Heart Disease',
                    labels={'sex': 'Gender', 'target': 'Heart Disease'},
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                    barmode='group'
                )
                fig.update_xaxis(ticktext=['Female', 'Male'], tickvals=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                gender_target_pct = pd.crosstab(heart_data['sex'], heart_data['target'], normalize='index') * 100
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='No Disease', x=['Female', 'Male'], y=gender_target_pct[0]))
                fig.add_trace(go.Bar(name='Disease', x=['Female', 'Male'], y=gender_target_pct[1]))
                fig.update_layout(
                    title='Gender Distribution by Target (%)',
                    barmode='stack',
                    yaxis_title='Percentage (%)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Chest Pain Analysis":
            st.markdown("### 💔 Chest Pain Type Analysis")
            
            cp_data = heart_data.groupby(['cp', 'target']).size().reset_index(name='count')
            cp_labels = ['Typical Angina', 'Atypical Angina', 'Non-Anginal', 'Asymptomatic']
            
            fig = px.bar(
                cp_data,
                x='cp',
                y='count',
                color='target',
                title='Chest Pain Type vs Heart Disease',
                labels={'cp': 'Chest Pain Type', 'target': 'Heart Disease'},
                color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                barmode='group'
            )
            fig.update_xaxis(ticktext=cp_labels, tickvals=[0, 1, 2, 3])
            st.plotly_chart(fig, use_container_width=True)
            
            # Percentage breakdown
            cp_pct = pd.crosstab(heart_data['cp'], heart_data['target'], normalize='index') * 100
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='No Disease', x=cp_labels, y=cp_pct[0]))
            fig.add_trace(go.Bar(name='Disease', x=cp_labels, y=cp_pct[1]))
            fig.update_layout(
                title='Chest Pain Type Distribution by Target (%)',
                barmode='stack',
                yaxis_title='Percentage (%)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Feature Relationships":
            st.markdown("### 🔗 Feature Relationships")
            
            col1, col2 = st.columns(2)
            
            with col1:
                feature_x = st.selectbox("Select X-axis:", ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
            
            with col2:
                feature_y = st.selectbox("Select Y-axis:", ['thalach', 'chol', 'trestbps', 'oldpeak', 'age'])
            
            fig = px.scatter(
                heart_data,
                x=feature_x,
                y=feature_y,
                color='target',
                title=f'{feature_x.upper()} vs {feature_y.upper()}',
                labels={'target': 'Heart Disease'},
                color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                opacity=0.6,
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "3D Analysis":
            st.markdown("### 🌐 3D Feature Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                feature_x = st.selectbox("X-axis:", ['age', 'trestbps', 'chol', 'thalach'])
            with col2:
                feature_y = st.selectbox("Y-axis:", ['chol', 'trestbps', 'thalach', 'oldpeak'])
            with col3:
                feature_z = st.selectbox("Z-axis:", ['thalach', 'oldpeak', 'age', 'chol'])
            
            fig = px.scatter_3d(
                heart_data,
                x=feature_x,
                y=feature_y,
                z=feature_z,
                color='target',
                title=f'3D: {feature_x.upper()} vs {feature_y.upper()} vs {feature_z.upper()}',
                labels={'target': 'Heart Disease'},
                color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                opacity=0.7
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## 🫀 Heart Disease Prediction System
    
    ### 📋 Project Information
    - **Author:** AyushMaurya13
    - **Date:** 2025-11-01 20:35:32
    - **Version:** 2.0
    - **Framework:** Streamlit + Scikit-learn
    
    ### 🎯 Objectives
    This project aims to predict the likelihood of heart disease using machine learning techniques.
    The system analyzes 13 clinical features to provide accurate predictions and risk assessments.
    
    ### 🔬 Methodology
    
    1. **Data Collection & Preprocessing**
       - 303 patient records
       - 13 clinical features
       - StandardScaler normalization
       - Train-test split (80-20)
    
    2. **Model Development**
       - 8 ML algorithms compared
       - Cross-validation (5-fold)
       - Hyperparameter tuning (GridSearchCV)
       - Best model selection
    
    3. **Evaluation Metrics**
       - Accuracy
       - Precision & Recall
       - F1-Score
       - ROC-AUC
       - Confusion Matrix
    
    ### 📊 Features Used
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Clinical Measurements:**
        - Age
        - Sex
        - Resting Blood Pressure
        - Serum Cholesterol
        - Fasting Blood Sugar
        - Maximum Heart Rate
        """)
    
    with col2:
        st.markdown("""
        **Diagnostic Tests:**
        - Chest Pain Type
        - Resting ECG Results
        - Exercise Induced Angina
        - ST Depression (Oldpeak)
        - Slope of ST Segment
        - Number of Major Vessels
        - Thalassemia
        """)
    
    st.markdown("""
    ### 🤖 Machine Learning Models
    
    The following algorithms were evaluated:
    1. Logistic Regression
    2. Random Forest Classifier
    3. Gradient Boosting Classifier
    4. Support Vector Machine (SVM)
    5. K-Nearest Neighbors (KNN)
    6. Decision Tree Classifier
    7. Naive Bayes
    8. AdaBoost Classifier
    
    ### 📈 Performance
    
    - **Best Model:** Random Forest / Gradient Boosting
    - **Accuracy:** 85%+
    - **ROC-AUC:** 0.90+
    - **Cross-Validation Score:** 0.84+
    
    ### ⚠️ Disclaimer
    
    This tool is for educational and research purposes only. It should **NOT** be used as a substitute 
    for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
    professionals for medical concerns.
    
    ### 📚 References
    
    - UCI Machine Learning Repository - Heart Disease Dataset
    - Scikit-learn Documentation
    - Streamlit Documentation
    
    
    ---
    
    **Built with using Streamlit and Scikit-learn**
    """)
    
    # Display system info
    st.markdown("### 💻 System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Python Version**\n{st.__version__}")
    with col2:
        st.info(f"**Streamlit Version**\n{st.__version__}")
    with col3:
        st.info(f"**Current Time (UTC)**\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>🫀 Heart Disease Prediction System </p>
    </div>
    """,
    unsafe_allow_html=True
)