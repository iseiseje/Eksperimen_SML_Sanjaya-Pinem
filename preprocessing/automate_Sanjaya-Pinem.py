import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    print(f"Loading data from: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):
    print("Memulai pipeline preprocessing...")
    
    # 1. Menghapus data duplikat
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. Imputasi nilai kosong
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)
    
    # 3. Mencegah Data Leakage
    columns_to_drop = ['reservation_status', 'reservation_status_date']
    df_clean = df.drop(columns=columns_to_drop)
    
    # 4. Pemisahan Fitur dan Target
    target_column = 'is_canceled'
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    # 5. Feature Encoding
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        
    # 6. Feature Scaling
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_encoded)
    X_scaled = pd.DataFrame(X_scaled_array, columns=X_encoded.columns)
    
    # 7. Penggabungan akhir
    df_final = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    return df_final

def main():
    # Menggunakan path dinamis agar kompatibel dengan GitHub Actions
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "dataset_raw", "hotel_bookings.csv")
    save_dir = os.path.join(base_dir, "dataset_preprocessing")
    save_path = os.path.join(save_dir, "hotel_bookings_processed.csv")
    
    # Eksekusi Pipeline
    df = load_data(raw_path)
    df_processed = preprocess_data(df)
    
    # Menyimpan Output
    os.makedirs(save_dir, exist_ok=True)
    df_processed.to_csv(save_path, index=False)
    print(f"SUKSES: Data siap latih telah disimpan di {save_path}")

if __name__ == "__main__":
    main()

    # Trigger CI/CD pipeline