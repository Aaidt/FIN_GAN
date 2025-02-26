import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Recreate the same synthetic dataset
np.random.seed(42)  # Same seed for reproducibility
n_samples = 10000

# Create legitimate transactions
legitimate = np.random.normal(loc=0, scale=1, size=(n_samples, 10))
legitimate_labels = np.zeros(n_samples)

# Create fraudulent transactions
fraudulent = np.random.normal(loc=2, scale=2, size=(int(n_samples * 0.1), 10))
fraudulent_labels = np.ones(int(n_samples * 0.1))

# Combine the data
X = np.vstack([legitimate, fraudulent])
y = np.hstack([legitimate_labels, fraudulent_labels])

# Create DataFrame
feature_names = [f'V{i}' for i in range(1, 11)]
data = pd.DataFrame(X, columns=feature_names)
data['Class'] = y

# Display first few rows and basic information
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())
print("\nClass Distribution:")
print(data['Class'].value_counts())

# Visualize the distribution of features for both classes
plt.figure(figsize=(15, 6))
for i in range(3):  # Show first 3 features as example
    plt.subplot(1, 3, i+1)
    sns.kdeplot(data=data, x=f'V{i+1}', hue='Class')
    plt.title(f'Distribution of V{i+1}')
plt.tight_layout()
plt.show()

# Save dataset to CSV
data.to_csv('synthetic_fraud_data.csv', index=False)
print("\nDataset saved as 'synthetic_fraud_data.csv'")


# ------------------------------------------------------------------------------------------------------------
# # Generate a synthetic dataset for demonstration
# np.random.seed(42)
# n_samples = 10000

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the realistic dataset
data = pd.read_csv('realistic_financial_transactions.csv')

# Extract features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Handle categorical features
categorical_features = ['MerchantCategory', 'PaymentMethod', 'DeviceUsed']
numeric_features = ['Amount', 'DistanceFromHome', 'HoursSincePrevTransaction']
boolean_features = ['AuthenticationFailed', 'IsWeekend', 'IsNightTime']

# Create timestamp features
X['Hour'] = pd.to_datetime(X['Timestamp']).dt.hour
X['DayOfWeek'] = pd.to_datetime(X['Timestamp']).dt.dayofweek
numeric_features.extend(['Hour', 'DayOfWeek'])

# Drop original timestamp and ID
X = X.drop(['Timestamp', 'TransactionID'], axis=1)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features),
        ('bool', 'passthrough', boolean_features)
    ])

# Preprocess the data
X_processed = preprocessor.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
# ------------------------------------------------------------------------------------------------------------



# Create legitimate transactions
legitimate = np.random.normal(loc=0, scale=1, size=(n_samples, 10))
legitimate_labels = np.zeros(n_samples)

# Create fraudulent transactions (fewer, with different distribution)
fraudulent = np.random.normal(loc=2, scale=2, size=(int(n_samples * 0.1), 10))
fraudulent_labels = np.ones(int(n_samples * 0.1))

# Combine the data
X = np.vstack([legitimate, fraudulent])
y = np.hstack([legitimate_labels, fraudulent_labels])

# Create a DataFrame
feature_names = [f'V{i}' for i in range(1, 11)]
data = pd.DataFrame(X, columns=feature_names)
data['Class'] = y

# Preprocess the data
X = data.drop("Class", axis=1)
y = data["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# After preprocessing
import json
import numpy as np

training_config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
    "model_type": "GAN"
}


# Function to convert NumPy arrays to lists
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if 'training_config' not in locals():
    raise ValueError("Error: training_config is not defined. Check where it's supposed to be initialized.")
print("training_config:", training_config)

# Save the config file correctly
with open("training_config.json", "w") as f:
    json.dump(training_config, f, default=convert_numpy)

    
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Improved GAN Architecture
class FraudGAN:
    def __init__(self, input_dim, latent_dim=100):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Build the generator
        self.generator = self.build_generator()
        
        # Build the combined model
        self.discriminator.trainable = False
        noise = tf.keras.Input(shape=(self.latent_dim,))
        generated_data = self.generator(noise)
        validity = self.discriminator(generated_data)
        self.combined = tf.keras.Model(noise, validity)
        self.combined.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )

    def build_generator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_dim=self.latent_dim),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.input_dim, activation='tanh')
        ])
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, input_dim=self.input_dim),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def train(self, X_train, epochs=1000, batch_size=128, sample_interval=100):
        
         # Add model checkpointing
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'generator_best.keras',
            save_best_only=True,
            monitor='loss'
        )
        
        # Labels for real and fake data
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_data = X_train[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            generated_data = self.generator.predict(noise)
            
            d_loss_real = self.discriminator.train_on_batch(real_data, real)
            d_loss_fake = self.discriminator.train_on_batch(generated_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Save intermediate checkpoints
            if epoch % 500 == 0:
                self.generator.save(f'generator_epoch_{epoch}.keras')
            
            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, real)
            
            if epoch % sample_interval == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Initialize and train the GAN
gan = FraudGAN(input_dim=X_train.shape[1])
gan.train(X_train, epochs=1000)

# Generate synthetic fraud data
noise = np.random.normal(0, 1, (int(len(X_train) * 0.5), gan.latent_dim))
synthetic_fraud_data = gan.generator.predict(noise)

# Combine real and synthetic data
X_augmented = np.vstack([X_train, synthetic_fraud_data])
y_augmented = np.hstack([y_train, np.ones(len(synthetic_fraud_data))])

# Train and evaluate the fraud detection model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_augmented, y_augmented)

# Evaluate the model
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()  

# --------------------------------------------------------------------------------------
# To save the GAN model:
# After GAN training
gan.generator.save('fraud_generator_model.keras')  
gan.discriminator.save('fraud_discriminator_model.keras')

# To save the Random Forest model:
# After training the classifier
import joblib
joblib.dump(clf, 'fraud_detection_model.joblib')


# To Load and Use Saved Models:
# Load GAN models
loaded_generator = tf.keras.models.load_model('fraud_generator_model')

# Load Random Forest
loaded_clf = joblib.load('fraud_detection_model.joblib')

# Use the loaded models
# Generate synthetic data with loaded generator
new_synthetic_data = loaded_generator.predict(noise)

# Make predictions with loaded classifier
predictions = loaded_clf.predict(new_data)
# --------------------------------------------------------------------------------------

