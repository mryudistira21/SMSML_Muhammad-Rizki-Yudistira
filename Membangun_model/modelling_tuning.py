import os
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
import joblib

# Set tracking URI & experiment name
mlflow.set_tracking_uri("file:///C:/Users/GL65 RTX2070/Jupyter/SMSML_Muhammad-Rizki-Yudistira/Membangun_model/mlruns")
mlflow.set_experiment("adult-income-manual")

# Path dataset
base_path = "dataset_preprocessing"
X_train = pd.read_csv(os.path.join(base_path, "X_train.csv"))
X_test = pd.read_csv(os.path.join(base_path, "X_test.csv"))
y_train = pd.read_csv(os.path.join(base_path, "y_train.csv"))
y_test = pd.read_csv(os.path.join(base_path, "y_test.csv"))

# Grid parameter
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10]
}

# Loop kombinasi parameter
for params in ParameterGrid(param_grid):
    with mlflow.start_run():
        # Logging parameter
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])

        # Training model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )
        model.fit(X_train, y_train.values.ravel())

        # Evaluasi model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Logging metrik
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Simpan model ke file .pkl dan log sebagai artefak
        model_filename = f"model_rf_{params['n_estimators']}_{params['max_depth']}.pkl"
        joblib.dump(model, model_filename)
        mlflow.log_artifact(model_filename)
        os.remove(model_filename)

        print(f"✅ Run selesai: {params} | acc={acc:.4f}, f1={f1:.4f}")