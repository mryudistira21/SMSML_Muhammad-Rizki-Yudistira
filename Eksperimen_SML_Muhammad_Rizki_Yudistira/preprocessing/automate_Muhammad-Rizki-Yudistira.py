import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_creditcard(path_csv, test_size=0.2, random_state=42):
    """
    Fungsi untuk preprocessing dan menyimpan hasil ke CSV.
    """

    # Load dataset
    df = pd.read_csv(path_csv)

    # Hapus duplikat
    df = df.drop_duplicates()

    # Pisah fitur dan target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Scaling kolom Time dan Amount
    scaler = StandardScaler()
    X_scaled = X.copy()
    cols_to_scale = ["Time", "Amount"]
    X_scaled[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Simpan hasil ke folder output
    path_csv = "../dataset_raw/creditcard.csv"
    output_dir = "dataset_preprocessed"

    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print("✅ Dataset selesai diproses dan disimpan ke folder:", output_dir)

if __name__ == "__main__":
    preprocess_creditcard(
        path_csv=r"C:\\Users\\GL65 RTX2070\\Jupyter\\SMSML_Muhammad-Rizki-Yudistira\\Eksperimen_SML_Muhammad_Rizki_Yudistira\\dataset_raw\\creditcard.csv"
    )

