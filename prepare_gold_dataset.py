import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

# ===== 1. ดึงข้อมูลจาก MT5 =====
if not mt5.initialize():
    print("MT5 init failed:", mt5.last_error())
    quit()

symbol = "GOLD"
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1000)
mt5.shutdown()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# ===== 2. เตรียมชุดข้อมูล =====
def create_features_and_labels(df, window_size=50, lookahead=5, threshold=0.0005):
    closes = df['close'].values
    X, y = [], []

    for i in range(len(closes) - window_size - lookahead):
        window = closes[i:i + window_size]
        future = closes[i + window_size:i + window_size + lookahead]
        future_mean = np.mean(future)
        current = closes[i + window_size - 1]
        delta = (future_mean - current) / current

        if delta > threshold:
            label = 2  # ขึ้น
        elif delta < -threshold:
            label = 1  # ลง
        else:
            label = 0  # นิ่ง

        X.append(window)
        y.append(label)

    return np.array(X), np.array(y)

X, y = create_features_and_labels(df)

print("Shape X:", X.shape)
print("Shape y:", y.shape)
print("ตัวอย่าง label:", y[:10])
