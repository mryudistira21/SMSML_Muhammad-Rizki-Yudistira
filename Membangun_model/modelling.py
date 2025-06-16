import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Autolog aktif
mlflow.sklearn.autolog()

# Set tracking URI (jika belum diset di luar)
mlflow.set_tracking_uri("file:///C:/Users/GL65 RTX2070/Jupyter/SMSML_Muhammad-Rizki-Yudistira/Membangun_model/mlruns")
mlflow.set_experiment("adult-income-basic")

# Load dataset
base_path = "dataset_preprocessing"
X_train = pd.read_csv(f"{base_path}/X_train.csv")
X_test = pd.read_csv(f"{base_path}/X_test.csv")
y_train = pd.read_csv(f"{base_path}/y_train.csv")
y_test = pd.read_csv(f"{base_path}/y_test.csv")

# Training
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")