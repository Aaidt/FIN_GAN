import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = None

# Configure page
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Sidebar controls
st.sidebar.header("System Controls")
section = st.sidebar.radio("Navigation", [
    "Home",
    "Generate Data",
    "Fraud Detection",
    "Model Dashboard"
])

# Main content
def main():
    if section == "Home":
        st.title("Financial Fraud Detection System")
        st.markdown("""
        ### System Components:
        - **GAN-Powered Synthetic Data Generation**
        - **Real-Time Fraud Prediction**
        - **Model Performance Monitoring**
        """)
        
    elif section == "Generate Data":
        st.title("Synthetic Data Generation")
        
        # Model loading
        if st.button("Load Generator Model"):
            try:
                st.session_state.generator = tf.keras.models.load_model('fraud_generator_model.keras')
                st.success("GAN Generator loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
        
        # Data generation UI
        if hasattr(st.session_state, 'generator'):
            num_samples = st.slider("Number of samples", 100, 5000, 1000)
            if st.button("Generate Fraud Patterns"):
                noise = tf.random.normal([num_samples, 100])
                generated_data = st.session_state.generator(noise)
                st.session_state.generated_data = generated_data.numpy()
                st.success(f"Generated {num_samples} synthetic transactions!")
            
            if st.session_state.generated_data is not None:
                st.subheader("Generated Data Analysis")
                df = pd.DataFrame(st.session_state.generated_data)
                
                # Visualizations
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Feature Distributions")
                    fig1, ax1 = plt.subplots()
                    df.iloc[:, :5].plot(kind='kde', ax=ax1)
                    st.pyplot(fig1)
                
                with col2:
                    st.write("Correlation Matrix")
                    fig2, ax2 = plt.subplots()
                    sns.heatmap(df.corr(), ax=ax2)
                    st.pyplot(fig2)

    elif section == "Fraud Detection":
        st.title("Real-Time Fraud Prediction")
        
        try:
            clf = joblib.load('fraud_detection_model.joblib')
            scaler = StandardScaler()
            
            # Prediction interface
            with st.form("prediction_form"):
                st.subheader("Transaction Details")
                
                # Create dynamic input fields
                inputs = {}
                for i in range(10):
                    inputs[f'V{i+1}'] = st.number_input(f'Feature V{i+1}', value=0.0)
                
                if st.form_submit_button("Analyze"):
                    # Preprocess input
                    input_df = pd.DataFrame([inputs])
                    scaled_input = scaler.fit_transform(input_df)
                    
                    # Predict
                    proba = clf.predict_proba(scaled_input)[0][1]
                    
                    # Display results
                    st.subheader("Result")
                    if proba > 0.7:
                        st.error(f"High fraud risk ({proba:.2%})")
                    else:
                        st.success(f"Low risk ({proba:.2%})")
        
        except FileNotFoundError:
            st.warning("Fraud detection model not found!")

    elif section == "Model Dashboard":
        st.title("Model Performance Dashboard")
        
        # Load sample metrics (replace with your actual metrics)
        st.subheader("Classification Metrics")
        metrics = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'Value': [0.92, 0.88, 0.90, 0.94]
        })
        st.dataframe(metrics.style.highlight_max(axis=0))
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap([[950, 50], [20, 980]], 
                    annot=True, fmt='d', 
                    cmap='Blues',
                    ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

if __name__ == "__main__":
    main()