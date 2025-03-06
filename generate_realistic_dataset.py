import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random

def generate_dataset(n_samples=10000):
    # Your dataset generation logic here
    # Set seed for reproducibility
    np.random.seed(42)

    # Number of transactions to generate
    n_legitimate = 10000
    n_fraudulent = int(n_legitimate * 0.01)  # More realistic 1% fraud rate

    # Generate transaction timestamps over 3 months
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    dates_legitimate = [start_date + timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds()))
    ) for _ in range(n_legitimate)]

    # Fraudsters tend to operate in bursts
    fraud_start_dates = [
        datetime(2024, 1, 15),
        datetime(2024, 2, 10),
        datetime(2024, 3, 5)
    ]
    dates_fraudulent = []
    for _ in range(n_fraudulent):
        fraud_base = random.choice(fraud_start_dates)
        dates_fraudulent.append(fraud_base + timedelta(hours=random.randint(0, 72)))

    # Generate legitimate transaction amounts (lognormal distribution)
    amount_legitimate = np.random.lognormal(mean=4.0, sigma=1.0, size=n_legitimate)
    # Cap max amount and round to 2 decimal places
    amount_legitimate = np.minimum(amount_legitimate, 5000)
    amount_legitimate = np.round(amount_legitimate, 2)

    # Generate fraudulent amounts (different pattern - often small test transactions followed by large ones)
    amount_fraudulent = []
    for _ in range(n_fraudulent):
        if random.random() < 0.3:
            # Small "test" transactions
            amount = random.uniform(0.5, 10.0)
        else:
            # Larger fraudulent transactions
            amount = random.uniform(500, 3000)
        amount_fraudulent.append(round(amount, 2))

    # Merchant categories
    merchant_categories = [
        'Grocery', 'Restaurant', 'Gas', 'Online Shopping', 'Entertainment', 
        'Travel', 'Utilities', 'Healthcare', 'Electronics', 'Clothing'
    ]

    # Fraud is more common in certain categories
    fraud_prone_categories = ['Online Shopping', 'Electronics', 'Travel']
    merchant_legitimate = np.random.choice(merchant_categories, size=n_legitimate, 
                                        p=[0.25, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02])
    merchant_fraudulent = np.random.choice(merchant_categories, size=n_fraudulent,
                                        p=[0.05, 0.05, 0.05, 0.4, 0.05, 0.2, 0.05, 0.05, 0.1, 0.0])

    # Geographic location (distance from home in miles)
    # Legitimate transactions usually happen close to home
    distance_legitimate = np.random.exponential(scale=10, size=n_legitimate)
    distance_legitimate = np.round(distance_legitimate, 1)

    # Fraudulent transactions often happen far from home
    distance_fraudulent = np.random.choice([
        np.random.uniform(0, 5),     # Some happen locally (stolen physical card)
        np.random.uniform(500, 3000)  # Some happen far away (data breach)
    ], size=n_fraudulent, p=[0.2, 0.8])
    distance_fraudulent = np.round(distance_fraudulent, 1)

    # Payment method
    payment_methods = ['Credit Card', 'Debit Card', 'Mobile Wallet']
    payment_legitimate = np.random.choice(payment_methods, size=n_legitimate, p=[0.4, 0.4, 0.2])
    # Fraudsters prefer credit cards due to higher limits
    payment_fraudulent = np.random.choice(payment_methods, size=n_fraudulent, p=[0.8, 0.15, 0.05])

    # Previous transaction time difference (in hours)
    prev_time_diff_legitimate = np.random.exponential(scale=24, size=n_legitimate)
    prev_time_diff_legitimate = np.round(prev_time_diff_legitimate, 1)

    # Fraudulent transactions often happen in quick succession
    prev_time_diff_fraudulent = np.random.exponential(scale=2, size=n_fraudulent)
    prev_time_diff_fraudulent = np.round(prev_time_diff_fraudulent, 1)

    # Device used for transaction
    devices = ['Mobile App', 'Web Browser', 'In-Person', 'Phone']
    device_legitimate = np.random.choice(devices, size=n_legitimate, p=[0.3, 0.3, 0.35, 0.05])
    # Fraudsters often use web browsers
    device_fraudulent = np.random.choice(devices, size=n_fraudulent, p=[0.2, 0.7, 0.05, 0.05])

    # Transaction success rate
    # Generate normal authentication failure rate
    auth_fails_legitimate = np.random.choice([0, 1], size=n_legitimate, p=[0.95, 0.05])
    # Fraudsters often fail authentication
    auth_fails_fraudulent = np.random.choice([0, 1], size=n_fraudulent, p=[0.7, 0.3])

    # Create DataFrames for legitimate and fraudulent transactions
    legitimate_df = pd.DataFrame({
        'TransactionID': range(1, n_legitimate + 1),
        'Timestamp': dates_legitimate,
        'Amount': amount_legitimate,
        'MerchantCategory': merchant_legitimate,
        'DistanceFromHome': distance_legitimate,
        'PaymentMethod': payment_legitimate,
        'HoursSincePrevTransaction': prev_time_diff_legitimate,
        'DeviceUsed': device_legitimate,
        'AuthenticationFailed': auth_fails_legitimate,
        'IsWeekend': [d.weekday() >= 5 for d in dates_legitimate],
        'IsNightTime': [(d.hour >= 22 or d.hour <= 5) for d in dates_legitimate],
        'Class': 0  # Legitimate
    })

    fraudulent_df = pd.DataFrame({
        'TransactionID': range(n_legitimate + 1, n_legitimate + n_fraudulent + 1),
        'Timestamp': dates_fraudulent,
        'Amount': amount_fraudulent,
        'MerchantCategory': merchant_fraudulent,
        'DistanceFromHome': distance_fraudulent,
        'PaymentMethod': payment_fraudulent,
        'HoursSincePrevTransaction': prev_time_diff_fraudulent,
        'DeviceUsed': device_fraudulent,
        'AuthenticationFailed': auth_fails_fraudulent,
        'IsWeekend': [d.weekday() >= 5 for d in dates_fraudulent],
        'IsNightTime': [(d.hour >= 22 or d.hour <= 5) for d in dates_fraudulent],
        'Class': 1  # Fraudulent
    })

    # Add a new feature to the dataset
    # legitimate_df['NewFeature'] = np.random.normal(loc=0, scale=1, size=n_legitimate)
    # fraudulent_df['NewFeature'] = np.random.normal(loc=0, scale=1, size=n_fraudulent)
    legitimate_df['NewFeature'] = np.random.normal(loc=0.2, scale=0.5, size=n_legitimate)
    fraudulent_df['NewFeature'] = np.random.normal(loc=1.5, scale=0.8, size=n_fraudulent)



    # Combine and shuffle
    all_transactions = pd.concat([legitimate_df, fraudulent_df])
    all_transactions = all_transactions.sample(frac=1).reset_index(drop=True)

    # Convert timestamp to string for easier CSV storage
    all_transactions['Timestamp'] = all_transactions['Timestamp'].astype(str)

    # Display dataset information
    print("Dataset Overview:")
    print(f"Total transactions: {len(all_transactions)}")
    print(f"Legitimate transactions: {len(legitimate_df)}")
    print(f"Fraudulent transactions: {len(fraudulent_df)}")
    print(f"Fraud rate: {len(fraudulent_df)/len(all_transactions)*100:.2f}%")

    print("\nSample transactions:")
    print(all_transactions.head())

    # Plot distribution of key features
    plt.figure(figsize=(15, 12))

    plt.subplot(2, 2, 1)
    sns.histplot(data=all_transactions, x='Amount', hue='Class', bins=50, log_scale=True)
    plt.title('Transaction Amount Distribution')

    plt.subplot(2, 2, 2)
    sns.countplot(data=all_transactions, x='MerchantCategory', hue='Class')
    plt.title('Transactions by Merchant Category')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 3)
    sns.histplot(data=all_transactions, x='DistanceFromHome', hue='Class', bins=30, log_scale=True)
    plt.title('Distance From Home Distribution')

    plt.subplot(2, 2, 4)
    sns.countplot(data=all_transactions, x='DeviceUsed', hue='Class')
    plt.title('Device Used for Transaction')

    plt.tight_layout()
    plt.show()

    # Save the dataset to CSV
    all_transactions.to_csv('realistic_financial_transactions.csv', index=False)
    print("\nDataset saved as 'realistic_financial_transactions.csv'")
    
    return all_transactions
