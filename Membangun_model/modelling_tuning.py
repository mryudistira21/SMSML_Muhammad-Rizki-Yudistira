import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
import joblib
import os

# Path data
base_path = r"..\\Eksperimen_SML_Muhammad_Rizki_Yudistira\\preprocessing\\dataset_preprocessed"
X_train = pd.read_csv(f"{base_path}/X_train.csv")
X_test = pd.read_csv(f"{base_path}/X_test.csv")
y_train = pd.read_csv(f"{base_path}/y_train.csv")
y_test = pd.read_csv(f"{base_path}/y_test.csv")

# Set experiment
mlflow.set_tracking_uri("file:///C:/Users/GL65 RTX2070/Jupyter/SMSML_Muhammad-Rizki-Yudistira/Membangun_model/mlruns")
mlflow.set_experiment("creditcard-fraud-detection-manual")


# Grid parameter
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10]
}

for params in ParameterGrid(param_grid):
    with mlflow.start_run():
        # Logging manual param
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])

        # Train model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )
        model.fit(X_train, y_train.values.ravel())

        # Predict & metrics
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Logging manual metric
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Simpan model
        model_path = f"model_rf_{params['n_estimators']}_{params['max_depth']}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        os.remove(model_path)  # bersihkan file lokal

        print(f"✅ Run selesai: {params} | acc={acc:.4f}, f1={f1:.4f}")
