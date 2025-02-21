import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Load components (only once)
@st.cache_resource
def load_components():
    generator = tf.keras.models.load_model('fraud_generator_model')
    clf = joblib.load('fraud_detection_model.joblib')
    scaler = StandardScaler()
    return generator, clf, scaler

generator, clf, scaler = load_components()

# Sidebar controls
st.sidebar.header("Controls")
section = st.sidebar.radio("Navigation", [
    "System Overview",
    "Generate Synthetic Data",
    "Fraud Detection",
    "Model Performance",
    "Live Prediction"
])

# Main content area
if section == "System Overview":
    st.title("Financial Fraud Detection System")
    st.image("fraud_detection.jpg", use_column_width=True)
    st.markdown("""
    ### System Capabilities:
    1. Synthetic fraud data generation using GANs
    2. Real-time fraud prediction
    3. Model performance visualization
    4. Transaction pattern analysis
    """)

elif section == "Generate Synthetic Data":
    st.title("Synthetic Data Generation")
    
    col1, col2 = st.columns(2)
    with col1:
        num_samples = st.slider("Number of samples to generate", 100, 5000, 1000)
        noise = np.random.normal(0, 1, (num_samples, 100))
        if st.button("Generate Data"):
            synthetic_data = generator.predict(noise)
            synthetic_df = pd.DataFrame(synthetic_data, 
                                     columns=[f'V{i}' for i in range(1, 11)])
            synthetic_df['Class'] = np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])
            st.session_state.synthetic_data = synthetic_df
            
    if 'synthetic_data' in st.session_state:
        with col2:
            st.subheader("Generated Data Preview")
            st.dataframe(st.session_state.synthetic_data.head())
            
            # Visualization
            st.subheader("Data Distribution")
            feature = st.selectbox("Select feature to visualize", 
                                 st.session_state.synthetic_data.columns[:-1])
            
            fig, ax = plt.subplots()
            sns.kdeplot(data=st.session_state.synthetic_data, x=feature, hue='Class', ax=ax)
            st.pyplot(fig)

elif section == "Fraud Detection":
    st.title("Real-time Fraud Detection")
    
    with st.form("transaction_form"):
        st.subheader("Transaction Details")
        col1, col2 = st.columns(2)
        
        # Create input fields for all features
        transaction_data = {}
        for i in range(10):
            with col1 if i < 5 else col2:
                transaction_data[f'V{i+1}'] = st.number_input(f'V{i+1}', value=0.0)
        
        submitted = st.form_submit_button("Check Fraud Probability")
        
        if submitted:
            # Prepare input data
            input_df = pd.DataFrame([transaction_data])
            scaled_input = scaler.fit_transform(input_df)
            
            # Make prediction
            prediction = clf.predict(scaled_input)
            probability = clf.predict_proba(scaled_input)[0][1]
            
            # Display results
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error(f"Fraud Detected! (Probability: {probability:.2%})")
            else:
                st.success(f"Legitimate Transaction (Probability: {1-probability:.2%})")

elif section == "Model Performance":
    st.title("Model Performance Analysis")
    
    # Load sample evaluation results (replace with your actual metrics)
    st.subheader("Classification Metrics")
    st.write(pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
        'Value': [0.92, 0.88, 0.90, 0.95]
    }))
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap([[950, 50], [20, 980]], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

elif section == "Live Prediction":
    st.title("Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload transaction data (CSV)", type="csv")
    if uploaded_file:
        test_data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(test_data.head())
        
        if st.button("Run Fraud Detection"):
            # Preprocess data
            scaled_data = scaler.transform(test_data)
            
            # Make predictions
            predictions = clf.predict(scaled_data)
            probabilities = clf.predict_proba(scaled_data)[:, 1]
            
            # Add to dataframe
            results = test_data.copy()
            results['Fraud Probability'] = probabilities
            results['Prediction'] = ['Fraud' if p == 1 else 'Legitimate' for p in predictions]
            
            st.subheader("Detection Results")
            st.dataframe(results.sort_values('Fraud Probability', ascending=False))
            
            # Download button
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Results",
                data=csv,
                file_name='fraud_predictions.csv',
                mime='text/csv'
            )

# Add custom CSS
st.markdown("""
<style>
    .stNumberInput {width: 200px;}
    .st-bb {background-color: #f0f2f6;}
    .reportview-container {background: #f0f2f6;}
    h1 {color: #2a4a7e;}
    .css-1aumxhk {background-color: #ffffff;}
</style>
""", unsafe_allow_html=True)