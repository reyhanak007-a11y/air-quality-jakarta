import pandas as pd

# Load dataset sesuai format yang diberikan
df = pd.read_csv('dataset/ispu_dki_all.csv')

# Convert tanggal ke datetime
df['tanggal'] = pd.to_datetime(df['tanggal'])

# Extract bulan dan hari untuk fitur temporal
df['bulan'] = df['tanggal'].dt.month
df['hari'] = df['tanggal'].dt.day

# Ekstrak nama stasiun tanpa lokasi spesifik
df['nama_stasiun'] = df['stasiun'].str.extract(r'(DKI\d+)')[0]

# Hitung moving average untuk parameter polutan
for pollutant in ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']:
    df[f'{pollutant}_ma7'] = df.groupby('nama_stasiun')[pollutant].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

# Tambahkan fitur interaksi antara polutan dan parameter kritis
df['critical_encoded'] = df['critical'].map({
    'PM10': 0, 'PM25': 1, 'SO2': 2, 'CO': 3, 'O3': 4, 'NO2': 5
})

# Buat fitur kategorikal untuk model
df['kategori_encoded'] = df['categori'].map({
    'BAIK': 0, 'SEDANG': 1, 'TIDAK SEHAT': 2, 'SANGAT TIDAK SEHAT': 3, 'BERBAHAYA': 4
})