import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    """
    Fungsi memuat data dengan Path Absolut & Mencegah Data Leakage.
    """
    print("[INFO] Memuat data training dan testing...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    train_path = os.path.join(base_dir, "hotel_booking_preprocessing", "train_processed.csv")
    test_path = os.path.join(base_dir, "hotel_booking_preprocessing", "test_processed.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File CSV tidak ditemukan di: {train_path}\nPastikan Anda sudah meng-copy folder 'hotel_booking_preprocessing' ke dalam 'Membangun_model'.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    drop_cols = ['is_canceled']
    if 'reservation_status' in train.columns:
        drop_cols.append('reservation_status')
        
    X_train = train.drop(columns=drop_cols, errors='ignore')
    y_train = train['is_canceled']
    X_test = test.drop(columns=drop_cols, errors='ignore')
    y_test = test['is_canceled']
    
    print(f"[INFO] Data Siap. Ukuran Train: {X_train.shape}")
    return X_train, y_train, X_test, y_test

def train_basic_model(X_train, y_train, X_test, y_test):
    """
    Kriteria Basic: Melatih model dengan MLflow Autolog (Mode Turbo).
    """

    mlflow.set_tracking_uri("") 
    
    mlflow.autolog()
    
    print("[PROCESS] Melatih Model Random Forest (Basic)...")
    print("          (Mohon tunggu, sedang menggunakan seluruh core CPU...)")
    
    mlflow.set_experiment("Hotel_Booking_Basic_Kresna")
    
    with mlflow.start_run():
        # n_jobs=-1 : Pakai semua core CPU
        # verbose=1 : Tampilkan progress bar
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=1)
        
        # Training
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n[SUCCESS] Training Selesai! Akurasi Valid: {acc:.4f}")
        print("[INFO] Cek hasil detail di MLflow UI.")
        print("[INFO] Metrik, Parameter, dan Model sudah tersimpan otomatis oleh MLflow.")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    train_basic_model(X_train, y_train, X_test, y_test)