import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# == โหลดข้อมูล X, y (จาก prepare_gold_dataset.py) ==
from prepare_gold_dataset import X, y

# == Normalize ข้อมูล ==
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# == แบ่งชุดเทรน/ทดสอบ ==
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# == สร้าง MLP Model ==
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
model.fit(X_train, y_train)

# == ทำนายและประเมิน ==
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")

# == Confidence ของแต่ละการทำนาย ==
probs = model.predict_proba(X_test)
confidences = np.max(probs, axis=1)
print("\nความมั่นใจตัวอย่าง:", confidences[:10])

# == รายงานรายละเอียด ==
print("\n", classification_report(y_test, y_pred, target_names=["Sideways (0)", "Down (1)", "Up (2)"]))
