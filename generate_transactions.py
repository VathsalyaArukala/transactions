import pandas as pd
import random

data = []

for i in range(1000):
    sender = random.randint(1000,9999)
    receiver = random.randint(1000,9999)
    amount = random.randint(100,50000)
    txn_type = random.choice(['UPI','CARD','NETBANKING'])
    sh = random.choice([0,1])
    rh = random.choice([0,1])

    is_sender_fraud = sh
    is_receiver_fraud = rh
    is_fraud = 1 if sh==1 or rh==1 else 0

    data.append([sender,receiver,amount,txn_type,sh,rh,is_sender_fraud,is_receiver_fraud,is_fraud])

df = pd.DataFrame(data, columns=[
    'sender_id','receiver_id','amount','transaction_type',
    'sender_fraud_history','receiver_fraud_history',
    'is_sender_fraud','is_receiver_fraud','is_fraud'
])

df.to_csv("transactions.csv", index=False)
print("✅ Dataset generated!")