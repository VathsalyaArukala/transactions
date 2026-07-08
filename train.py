import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

os.makedirs("models", exist_ok=True)

np.random.seed(42)
n = 1000

data = pd.DataFrame({
    "sender_id": np.random.randint(1000, 9999, n),
    "receiver_id": np.random.randint(1000, 9999, n),
    "amount": np.random.randint(100, 10000, n),
    "transaction_type": np.random.choice(["UPI","CARD","NETBANKING"], n),
    "sender_fraud_history": np.random.randint(0,2,n),
    "receiver_fraud_history": np.random.randint(0,2,n)
})

# ✅ Separate targets
data["sender_target"] = data["sender_fraud_history"]
data["receiver_target"] = data["receiver_fraud_history"]

# Encode
le = LabelEncoder()
data["transaction_type"] = le.fit_transform(data["transaction_type"])

X = data[["sender_id","receiver_id","amount","transaction_type",
          "sender_fraud_history","receiver_fraud_history"]]

# Split for sender
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X, data["sender_target"], test_size=0.2)

# Split for receiver
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, data["receiver_target"], test_size=0.2)

# Models
sender_model = RandomForestClassifier()
receiver_model = RandomForestClassifier()

sender_model.fit(Xs_train, ys_train)
receiver_model.fit(Xr_train, yr_train)

# Save
joblib.dump(sender_model, "models/sender.pkl")
joblib.dump(receiver_model, "models/receiver.pkl")
joblib.dump(le, "models/encoder.pkl")

# Save test sets separately
joblib.dump(Xs_test, "models/Xs_test.pkl")
joblib.dump(ys_test, "models/ys_test.pkl")

joblib.dump(Xr_test, "models/Xr_test.pkl")
joblib.dump(yr_test, "models/yr_test.pkl")

print("✅ Training fixed and completed!")