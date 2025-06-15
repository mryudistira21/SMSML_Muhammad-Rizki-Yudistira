import pandas as pd
import mlflow
import os
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Autolog aktif
mlflow.sklearn.autolog()

# Load dataset dari folder preprocessing
base_path = os.path.join(os.path.dirname(__file__), "dataset_preprocessing")

X_train = pd.read_csv(os.path.join(base_path, "X_train.csv"))
X_test = pd.read_csv(os.path.join(base_path, "X_test.csv"))
y_train = pd.read_csv(os.path.join(base_path, "y_train.csv"))
y_test = pd.read_csv(os.path.join(base_path, "y_test.csv"))

# Training
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")