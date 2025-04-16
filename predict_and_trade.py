import MetaTrader5 as mt5

SYMBOL = "GOLD"
LOT = 0.1

def send_order(order_type):
    if not mt5.initialize():
        print("❌ MT5 init failed")
        return

    price = mt5.symbol_info_tick(SYMBOL).ask if order_type == "buy" else mt5.symbol_info_tick(SYMBOL).bid
    order = mt5.order_send(
        request={
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": LOT,
            "type": mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": f"MLP auto trade {order_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
    )
    mt5.shutdown()
    print("✅ ส่งคำสั่งแล้ว:", order)
