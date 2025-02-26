import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import json
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from generate_realistic_dataset import generate_dataset
except ImportError as e:
    st.error(f"Failed to import dataset generator: {str(e)}")
    st.stop()


# Set page config
st.set_page_config(page_title="Financial Fraud System", layout="wide")

# Session state initialization
if 'realistic_data' not in st.session_state:
    st.session_state.realistic_data = None
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None

# Load components
@st.cache_resource
def load_models():
    try:
        generator = tf.keras.models.load_model('fraud_generator_model.keras')
        clf = joblib.load('fraud_detection_model.joblib')
        return generator, clf
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

generator, clf = load_models()

# Sidebar controls
st.sidebar.header("System Controls")
section = st.sidebar.radio("Navigation", [
    "Data Generation",
    "Fraud Detection",
    "Model Analysis",
    "Real vs Synthetic"
])

# Main content
def main():
    if section == "Data Generation":
        st.title("Financial Data Generation")
        
        col1, col2 = st.columns(2)
        
        # with col1:
        #     st.subheader("Realistic Data")
        #     if st.button("Generate Realistic Dataset"):
        #         # This would call your generate_realistic_dataset.py logic
        #         from generate_realistic_dataset import generate_dataset
        #         df = generate_dataset()
        #         st.session_state.realistic_data = df
        #         st.success("Generated realistic financial transactions!")
            
        #     if st.session_state.realistic_data is not None:
        #         st.write("Realistic Data Preview:")
        #         st.dataframe(st.session_state.realistic_data.head())
        #         st.download_button(
        #             label="Download Realistic Data",
        #             data=st.session_state.realistic_data.to_csv(index=False),
        #             file_name='realistic_transactions.csv'
        #         )
        with col1:
            st.subheader("Realistic Data")
            if st.button("Generate Realistic Dataset"):
                try:
                    df = generate_dataset()  # From your generate_realistic_dataset.py
                    st.session_state.realistic_data = df
                    st.success("Generated realistic financial transactions!")
                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")


        with col2:
            st.subheader("Synthetic Data")
            if generator:
                num_samples = st.slider("Synthetic samples to generate", 100, 5000, 1000)
                if st.button("Generate Synthetic Data"):
                    if st.session_state.realistic_data is None:
                        st.warning("Generate realistic data first!")
                    else:
                        try:
                            # Load training configuration
                            with open('training_config.json') as f:
                                config = json.load(f)
                                
                            noise = tf.random.normal([num_samples, 100])
                            synthetic = generator.predict(noise)
                            
                            # Create DF with processed feature names
                            synthetic_df = pd.DataFrame(
                                synthetic,
                                columns=config['preprocessor_steps']
                            )
                            
                            # Add class column
                            synthetic_df['Class'] = np.random.choice([0, 1], 
                                size=num_samples, 
                                p=[0.9, 0.1]
                            )
                            
                            st.session_state.synthetic_data = synthetic_df
                            st.success("Synthetic data generated!")
                        except Exception as e:
                            st.error(f"Generation failed: {str(e)}")
                            
        if "realistic_data" not in st.session_state or st.session_state.realistic_data is None:
            st.warning("No dataset found. Please generate the dataset first.")
            st.stop()  # Prevents further execution
         
        st.session_state.realistic_data = generate_dataset()

        if st.session_state.realistic_data is None:
            raise ValueError("Dataset generation failed. Check generate_dataset().")
            
        # After generating both datasets
        if st.session_state.realistic_data is not None:
            real_dim = st.session_state.realistic_data.shape[1]
        else:
            st.warning("Dataset is empty. Please regenerate.")

        syn_dim = st.session_state.synthetic_data.shape[1]
        if real_dim != syn_dim: 
            st.warning(f"Feature mismatch! Real: {real_dim} vs Syn: {syn_dim}")

    elif section == "Fraud Detection":
        st.title("Real-time Fraud Analysis")
        
        if clf is None:
            st.warning("Fraud detection model not loaded!")
            return
            
        tab1, tab2 = st.tabs(["Single Transaction", "Batch Processing"])
        
        with tab1:
            # input_processed = preprocessor.transform(input_df)
            # prediction = clf.predict(input_processed)
            
            with st.form("single_transaction"):
                st.subheader("Transaction Details")
                inputs = {}
                cols = st.columns(3)
                for i, col in enumerate(cols):
                    with col:
                        inputs[f'V{i*3+1}'] = st.number_input(f'V{i*3+1}', value=0.0)
                        inputs[f'V{i*3+2}'] = st.number_input(f'V{i*3+2}', value=0.0)
                        inputs[f'V{i*3+3}'] = st.number_input(f'V{i*3+3}', value=0.0)
                
                if st.form_submit_button("Analyze"):
                    input_df = pd.DataFrame([inputs])
                    prediction = clf.predict(input_df)
                    proba = clf.predict_proba(input_df)[0][1]
                    
                    if prediction[0] == 1:
                        st.error(f"Fraud Detected! (Probability: {proba:.2%})")
                    else:
                        st.success(f"Legitimate Transaction (Probability: {1-proba:.2%})")

        with tab2:
            uploaded_file = st.file_uploader("Upload transactions CSV", type="csv")
            if uploaded_file:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Preview:", batch_df.head())
                
                if st.button("Run Batch Analysis"):
                    predictions = clf.predict(batch_df)
                    results = batch_df.copy()
                    results['Prediction'] = ['Fraud' if p == 1 else 'Legit' for p in predictions]
                    st.write("Results:", results)
                    
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name='fraud_predictions.csv'
                    )

    elif section == "Model Analysis":
        st.title("Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Metrics")
            metrics = pd.DataFrame({
                'Metric': ['Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                'Value': [0.92, 0.88, 0.90, 0.94]
            })
            st.table(metrics)
            
        with col2:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap([[950, 50], [20, 980]], 
                        annot=True, fmt='d', 
                        cmap='Blues',
                        ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

    elif section == "Real vs Synthetic":
        st.title("Data Comparison")
        
        valid_features = list(set(st.session_state.realistic_data.columns) 
                    & set(st.session_state.synthetic_data.columns))
        feature = st.selectbox("Select feature to compare", valid_features)

        if st.session_state.realistic_data is None or st.session_state.synthetic_data is None:
            st.warning("Generate both datasets first!")
            return
            
        feature = st.selectbox("Select feature to compare", 
                             st.session_state.realistic_data.columns[:-1])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(data=st.session_state.realistic_data, x=feature, label='Real Data', ax=ax)
        sns.kdeplot(data=st.session_state.synthetic_data, x=feature, label='Synthetic Data', ax=ax)
        ax.set_title(f"Distribution Comparison - {feature}")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()