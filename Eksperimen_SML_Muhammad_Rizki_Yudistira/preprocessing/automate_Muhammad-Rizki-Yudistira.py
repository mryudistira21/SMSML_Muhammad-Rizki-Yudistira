import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_adult_income(path_csv, test_size=0.2, random_state=42):
    """
    Preprocessing dataset Adult Income dan menyimpan hasil split ke CSV.
    """

    # Load dataset
    df = pd.read_csv(path_csv)

    # Hapus duplikat
    df = df.drop_duplicates()

    # Mapping target
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

    # Hapus baris dengan "?" di kolom kategorikal
    df = df[~df.isin(["?"]).any(axis=1)]

    # Pisah fitur dan target
    X = df.drop("income", axis=1)
    y = df["income"]

    # One-hot encoding kolom kategorikal
    X_encoded = pd.get_dummies(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Normalisasi kolom numerik
    num_cols = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Simpan ke folder preprocessing
    preprocessed_dir = r"C:\Users\GL65 RTX2070\Jupyter\SMSML_Muhammad-Rizki-Yudistira\Eksperimen_SML_Muhammad_Rizki_Yudistira\preprocessing\dataset_preprocessed"
    os.makedirs(preprocessed_dir, exist_ok=True)

    X_train.to_csv(os.path.join(preprocessed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(preprocessed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(preprocessed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(preprocessed_dir, "y_test.csv"), index=False)

    print("✅ Dataset disimpan ke folder preprocessing.")

    # Sekaligus salin ke folder Membangun_model/dataset_preprocessing
    model_dir = r"C:\Users\GL65 RTX2070\Jupyter\SMSML_Muhammad-Rizki-Yudistira\Membangun_model\dataset_preprocessing"
    os.makedirs(model_dir, exist_ok=True)

    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        src = os.path.join(preprocessed_dir, fname)
        dst = os.path.join(model_dir, fname)
        shutil.copy(src, dst)
        print(f"📁 Disalin ke folder Membangun_model: {fname}")

if __name__ == "__main__":
    preprocess_adult_income(
        path_csv=r"C:\Users\GL65 RTX2070\Jupyter\SMSML_Muhammad-Rizki-Yudistira\Eksperimen_SML_Muhammad_Rizki_Yudistira\dataset_raw\adult.csv"
    )
