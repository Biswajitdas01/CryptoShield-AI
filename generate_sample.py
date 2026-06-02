"""
Generate a realistic synthetic cryptocurrency transaction dataset for CryptoShield AI.
Run this script once to create data/crypto_transactions.csv
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import string

np.random.seed(42)
random.seed(42)

N_LEGIT = 4000
N_FRAUD = 800
N = N_LEGIT + N_FRAUD

def random_wallet():
    return "0x" + "".join(random.choices(string.hexdigits[:16], k=40))

wallets = [random_wallet() for _ in range(800)]
base_time = datetime(2023, 1, 1)

def gen_legitimate(n):
    rows = []
    for _ in range(n):
        ts = base_time + timedelta(seconds=random.randint(0, 365*86400))
        amount = np.random.lognormal(mean=3.5, sigma=1.2)
        gas = np.random.lognormal(mean=2.2, sigma=0.5)
        rows.append({
            "transaction_id": "TX" + "".join(random.choices(string.digits, k=10)),
            "sender_wallet": random.choice(wallets),
            "receiver_wallet": random.choice(wallets),
            "timestamp": ts,
            "amount_usd": round(amount, 4),
            "gas_fee_usd": round(gas * 0.01, 4),
            "transaction_speed_s": int(np.random.exponential(scale=30)),
            "num_transactions_24h": int(np.random.poisson(lam=5)),
            "wallet_age_days": int(np.random.exponential(scale=400)) + 1,
            "unique_receivers_30d": int(np.random.poisson(lam=8)),
            "avg_transaction_amount": round(np.random.lognormal(mean=3.5, sigma=1.0), 4),
            "std_transaction_amount": round(np.random.lognormal(mean=2.0, sigma=0.8), 4),
            "night_transaction": int(ts.hour < 6 or ts.hour > 22),
            "weekend_transaction": int(ts.weekday() >= 5),
            "cross_chain": int(random.random() < 0.05),
            "mixing_service_flag": 0,
            "rapid_movement_flag": 0,
            "blacklist_interaction": 0,
            "contract_interaction": int(random.random() < 0.3),
            "token_type": random.choice(["ETH", "BTC", "USDT", "BNB", "USDC"]),
            "network": random.choice(["Ethereum", "Binance", "Polygon"]),
            "is_fraud": 0,
        })
    return rows

def gen_fraud(n):
    rows = []
    for _ in range(n):
        ts = base_time + timedelta(seconds=random.randint(0, 365*86400))
        amount = np.random.choice([
            np.random.lognormal(mean=7, sigma=1.5),   # large amounts
            np.random.uniform(0.001, 0.01),            # dust attacks
        ])
        gas = np.random.uniform(0.001, 0.5)
        rows.append({
            "transaction_id": "TX" + "".join(random.choices(string.digits, k=10)),
            "sender_wallet": random.choice(wallets[:100]),  # concentrated senders
            "receiver_wallet": random_wallet(),
            "timestamp": ts,
            "amount_usd": round(amount, 4),
            "gas_fee_usd": round(gas, 4),
            "transaction_speed_s": int(np.random.exponential(scale=3)),   # very fast
            "num_transactions_24h": int(np.random.poisson(lam=45)),       # many txns
            "wallet_age_days": int(np.random.exponential(scale=15)) + 1,  # new wallets
            "unique_receivers_30d": int(np.random.poisson(lam=60)),       # many receivers
            "avg_transaction_amount": round(np.random.lognormal(mean=6, sigma=2), 4),
            "std_transaction_amount": round(np.random.lognormal(mean=5, sigma=1.5), 4),
            "night_transaction": int(random.random() < 0.7),
            "weekend_transaction": int(random.random() < 0.6),
            "cross_chain": int(random.random() < 0.5),
            "mixing_service_flag": int(random.random() < 0.7),
            "rapid_movement_flag": int(random.random() < 0.8),
            "blacklist_interaction": int(random.random() < 0.4),
            "contract_interaction": int(random.random() < 0.8),
            "token_type": random.choice(["ETH", "USDT", "BNB"]),
            "network": random.choice(["Ethereum", "Binance", "Polygon", "Arbitrum"]),
            "is_fraud": 1,
        })
    return rows

legit = gen_legitimate(N_LEGIT)
fraud = gen_fraud(N_FRAUD)
all_rows = legit + fraud
random.shuffle(all_rows)

df = pd.DataFrame(all_rows)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

output = "data/crypto_transactions.csv"
df.to_csv(output, index=False)
print(f"✅ Dataset saved: {output}")
print(f"   Total rows  : {len(df)}")
print(f"   Fraud rows  : {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
print(f"   Columns     : {list(df.columns)}")
