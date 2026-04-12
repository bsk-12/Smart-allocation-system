# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# =========================
# 2. LOAD DATASET
# =========================
df = pd.read_csv("/content/renamed_delivery_dataset.csv")  # use your file name
df.head()
print("✅ Dataset Loaded")
print(df.head())

# =========================
# 3. CLEAN COLUMN NAMES
# =========================
df.columns = df.columns.str.strip()

# =========================
# 4. ENCODE CATEGORICAL DATA
# =========================
label_encoders = {}

categorical_cols = [
    "delivery_partner",
    "package_type",
    "vehicle_type",
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("✅ Encoding Done")

# =========================
# 5. DEFINE FEATURES & TARGET
# =========================
X = df.drop(["delay"], axis=1)
y = df["delay"]

# =========================
# 6. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 7. TRAIN MODEL (REGRESSION)
# =========================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model Trained")

# =========================
# 8. EVALUATE MODEL
# =========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Mean Absolute Error: {mae:.2f}")
print(f"📈 R2 Score: {r2:.2f}")

# =========================
# 9. SAVE MODEL & ENCODERS
# =========================
joblib.dump(model, "diamond_price_model.pkl")
joblib.dump(label_encoders, "encoders.pkl")

print("✅ Model & Encoders Saved")
