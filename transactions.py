import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

data = {
    'transaction_id': range(1, n+1),
    'sender_id': np.random.randint(1000, 2000, n),
    'receiver_id': np.random.randint(2000, 3000, n),
    'amount': np.round(np.random.uniform(10, 10000, n), 2),
    'transaction_type': np.random.choice(['transfer', 'payment', 'withdrawal', 'deposit'], n),
    'sender_fraud_history': np.random.choice([0, 1], n, p=[0.95, 0.05]),
    'receiver_fraud_history': np.random.choice([0, 1], n, p=[0.98, 0.02]),
}

data['is_fraud'] = np.where(
    (data['amount'] > 8000) | (data['sender_fraud_history'] == 1) | (data['receiver_fraud_history'] == 1),
    1, 0
)

df = pd.DataFrame(data)
df.to_csv('transactions.csv', index=False)
print("transactions.csv generated")
