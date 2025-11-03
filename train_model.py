import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

print("\n🚦 Training Optimized Traffic Model — Target: 90%+ Accuracy\n")

# Load dataset
data = pd.read_csv(
    r"C:\Users\KIIT\Downloads\ai_traffic_optimizer-main (1)\ai_traffic_optimizer-main\traffic_log.csv"
)
print("✅ Loaded data shape:", data.shape)

# --- Feature Engineering ---
data["TotalTraffic"] = data[["North", "East", "South", "West"]].sum(axis=1)
for direction in ["North", "East", "South", "West"]:
    data[f"{direction}_Ratio"] = data[direction] / (data["TotalTraffic"] + 1e-5)

# Congestion index: how uneven the flow is
data["CongestionIndex"] = data[["North", "East", "South", "West"]].std(axis=1)

# Features and label
X = data.drop(columns=["GreenLight"])
y = data["GreenLight"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- Improved Gradient Boosting Model ---
model = HistGradientBoostingClassifier(
    max_iter=800,           # more boosting rounds
    learning_rate=0.04,     # smaller step for finer convergence
    max_depth=16,           # deeper trees capture complex rules
    l2_regularization=0.02, # slight regularization
    min_samples_leaf=10,    # smoother decision boundaries
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"🎯 Model Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=encoder.classes_))

# Save components
joblib.dump(model, "traffic_ai_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n💾 Model, encoder, and scaler saved successfully!")
