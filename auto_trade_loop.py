import time
import csv
from datetime import datetime
import numpy as np
import MetaTrader5 as mt5
import requests

from prepare_gold_dataset import df
from train_mlp_softmax import model, scaler
from predict_and_trade import send_order

CONFIDENCE_THRESHOLD = 0.90
SYMBOL = "GOLD"
DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1359747855082061965/Qqy83mwPrJnlQTEvSBhO9UY8xwfn1OREdpLCXJKojHikKb05gUX2qbnm6-GhzKiCSl-5"

def log_trade(prediction_label, confidence, action, price="-"):
    with open("trade_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            prediction_label,
            f"{confidence:.4f}",
            action,
            price
        ])

def notify_discord(message):
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
    except Exception as e:
        print("❌ แจ้งเตือน Discord ล้มเหลว:", e)

def predict_and_trade():
    print(f"\\n🔁 [{datetime.now().strftime('%H:%M:%S')}] ตรวจสอบสัญญาณเทรด...")

    df_updated = df.tail(1000).copy()
    closes = df_updated['close'].values
    if len(closes) < 50:
        print("❌ ข้อมูลไม่พอ")
        return

    latest = closes[-50:].reshape(1, -1)
    latest_scaled = scaler.transform(latest)

    probs = model.predict_proba(latest_scaled)
    prediction = np.argmax(probs)
    confidence = np.max(probs)

    label_map = ["sideways", "down", "up"]
    prediction_label = label_map[prediction]

    print(f"📊 ทำนาย: {prediction_label} ({confidence:.2%})")

    if confidence >= CONFIDENCE_THRESHOLD:
        if not mt5.initialize():
            print("❌ เชื่อมต่อ MT5 ไม่ได้")
            log_trade(prediction_label, confidence, "mt5_error")
            notify_discord("❌ Failed to connect to MT5")
            return

        if prediction == 2:
            price = mt5.symbol_info_tick(SYMBOL).ask
            send_order("buy")
            log_trade(prediction_label, confidence, "buy", price)
            notify_discord(f"📈 BUY GOLD\\nPrediction: {prediction_label}\\nConfidence: {confidence:.2%}\\nPrice: {price}")
        elif prediction == 1:
            price = mt5.symbol_info_tick(SYMBOL).bid
            send_order("sell")
            log_trade(prediction_label, confidence, "sell", price)
            notify_discord(f"📉 SELL GOLD\\nPrediction: {prediction_label}\\nConfidence: {confidence:.2%}\\nPrice: {price}")
        else:
            print("🟡 ทำนายว่านิ่ง — ไม่เทรด")
            log_trade(prediction_label, confidence, "no_trade_niti")
            notify_discord(f"🟡 Sideways — no trade\\nPrediction: {prediction_label}\\nConfidence: {confidence:.2%}")

        mt5.shutdown()
    else:
        print("⚠️ มั่นใจไม่พอ — งดเทรด")
        log_trade(prediction_label, confidence, "no_trade_low_conf")
        notify_discord(f"⚠️ No trade — low confidence ({confidence:.2%}) on prediction: {prediction_label}")

if __name__ == "__main__":
    print("🚀 เริ่มระบบ Auto Trade ทุก 5 นาที")
    while True:
        try:
            predict_and_trade()
            print("🕒 รอรอบถัดไปในอีก 5 นาที...")
        except Exception as e:
            print("❌ เกิดข้อผิดพลาด:", e)
        time.sleep(300)