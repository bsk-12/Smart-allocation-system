# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from google.colab import files
import joblib
joblib.dump(label_encoders, "encoders.pkl")
files.download("resource_model.pkl")
from google.colab import files
files.download("encoders.pkl")
# =========================
# 2. LOAD DATASET
# =========================
df = pd.read_csv("resource_allocation_dataset.csv")

print("✅ Dataset Loaded")
print(df.head())

# =========================
# 3. ENCODE CATEGORICAL DATA
# =========================
label_encoders = {}

categorical_cols = [
    "skill_level",
    "location",
    "task_type",
    "task_complexity",
    "required_skill"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le   # Save encoder for later use

print("✅ Encoding Done")

# =========================
# 4. DEFINE FEATURES & TARGET
# =========================
X = df.drop(["resource_id", "match"], axis=1)
y = df["match"]

# =========================
# 5. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. TRAIN MODEL
# =========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model Trained")

# =========================
# 7. EVALUATE MODEL
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.2f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 8. SAVE MODEL (IMPORTANT)
# =========================
import joblib

joblib.dump(model, "resource_model.pkl")

print("✅ Model Saved as resource_model.pkl")