from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ================================
# HASIL TRAINING SESUAI MODEL NYATA
# ================================

training_results = {
    "accuracy": 0.9736,  # dari output model kamu

    # Weighted average hasil classification report
    "precision": 0.97,
    "recall": 0.97,
    "f1": 0.97,

    "confusion_matrix": [
        [62, 9, 0],   # Aktual: BAIK
        [24, 311, 7], # Aktual: SEDANG
        [0, 3, 37]    # Aktual: TIDAK SEHAT
    ],

    "model_params": {
        "criterion": "entropy",
        "max_depth": 10,
        "min_samples_leaf": 1,
        "min_samples_split": 5
    },

    # feature importance kamu jadikan dict
    "feature_importance": {
        "pm25": 0.527433,
        "kelembaban": 0.189455,
        "no2": 0.115669,
        "bulan": 0.051264,
        "hari_minggu": 0.048787,
        "o3": 0.038289,
        "pm10": 0.014396,
        "co": 0.010441,
        "so2": 0.004267,
        "suhu": 0.0,
        "heat_index": 0.0,
        "kecepatan_angin": 0.0,
        "hari": 0.0,
        "jam": 0.0
    },

    "training_time": "10.93 detik"
}


# ==================================
# INISIALISASI FLASK
# ==================================

app = Flask(__name__)

MODEL_DIR = 'model_artifacts'

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'air_quality_dt_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    with open(os.path.join(MODEL_DIR, 'model_features.txt'), 'r') as f:
        features = [line.strip() for line in f.readlines()]

    print("Model dan artifacts berhasil dimuat.")
    print("Features:", features)

except Exception as e:
    print(f"Error memuat model: {e}")
    model = None
    features = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2', 'suhu', 'kelembaban', 'kecepatan_angin']


# ==============================
# ROUTE HALAMAN UTAMA
# ==============================

@app.route('/')
def home():
    return render_template('index.html', features=features)


# ==============================
# ROUTE PREDIKSI
# ==============================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}

        # Ambil input user
        for feature in features:
            if feature in ['heat_index', 'bulan', 'hari', 'jam', 'hari_minggu']:
                continue 

            value = request.form.get(feature)

            if not value or value.strip() == '':
                return render_template('index.html',
                                       error=f"Silakan isi nilai untuk {feature.replace('_', ' ')}",
                                       features=features)

            try:
                input_data[feature] = float(value)
            except:
                return render_template('index.html',
                                       error=f"Nilai untuk {feature} harus berupa angka.",
                                       features=features)

        # Hitung heat index jika tersedia
        if 'suhu' in input_data and 'kelembaban' in input_data:
            T = input_data['suhu']
            H = input_data['kelembaban']
            heat_index = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (H * 0.094))
            input_data['heat_index'] = heat_index

        # Tambah fitur waktu otomatis
        now = datetime.now()
        if 'bulan' in features: input_data['bulan'] = now.month
        if 'hari' in features: input_data['hari'] = now.day
        if 'jam' in features: input_data['jam'] = now.hour
        if 'hari_minggu' in features: input_data['hari_minggu'] = now.weekday()

        # Convert ke DataFrame
        input_df = pd.DataFrame([{f: input_data.get(f, 0) for f in features}])

        # ===================
        # MODE PREDIKSI NYATA
        # ===================
        if model is not None:
            scaled = scaler.transform(input_df)
            pred = model.predict(scaled)[0]
            proba = model.predict_proba(scaled)[0]

            category = label_encoder.inverse_transform([pred])[0]
            confidence = max(proba) * 100

            recommendations = get_health_recommendations(category)

            formatted_input = {k.replace('_', ' ').title(): v for k, v in input_data.items()}

            return render_template('result.html',
                                   category=category,
                                   confidence=confidence,
                                   recommendations=recommendations,
                                   input_data=formatted_input)

        # =================
        # MODE DEMO (NO MODEL)
        # =================
        else:
            demo_category = "SEDANG"
            demo_reco = get_health_recommendations(demo_category)

            return render_template('result.html',
                                   category=demo_category,
                                   confidence=85.0,
                                   recommendations=demo_reco,
                                   input_data=input_data)

    except Exception as e:
        print("Error prediksi:", e)
        return render_template('index.html',
                               error="Terjadi kesalahan saat memproses prediksi.",
                               features=features)


# ==============================
# ROUTE TRAINING RESULTS
# ==============================

@app.route('/training-results')
def training_results_page():
    return render_template('training_results.html', results=training_results)


# ==============================
# FUNGSI REKOMENDASI
# ==============================

def get_health_recommendations(category):
    recommendations = {
        'BAIK': [
            "Kualitas udara sangat baik untuk aktivitas luar ruangan.",
            "Aman bagi semua kelompok usia.",
            "Tidak diperlukan tindakan perlindungan khusus."
        ],
        'SEDANG': [
            "Kualitas udara cukup baik.",
            "Kelompok sensitif sebaiknya mengurangi aktivitas berat di luar.",
            "Umum aman beraktivitas normal."
        ],
        'TIDAK SEHAT': [
            "Gunakan masker saat berada di luar ruangan.",
            "Kurangi aktivitas luar ruangan terutama bagi kelompok rentan.",
            "Pastikan ruangan memiliki ventilasi baik."
        ],
        'SANGAT TIDAK SEHAT': [
            "Hindari aktivitas luar ruangan.",
            "Gunakan masker N95 bila harus keluar.",
            "Kelompok rentan harus tetap berada di dalam ruangan."
        ],
        'BERBAHAYA': [
            "KEADAAN DARURAT - Hindari semua aktivitas luar ruangan.",
            "Gunakan air purifier atau filtrasi udara.",
            "Segera cari bantuan medis jika muncul gejala berat."
        ]
    }
    return recommendations.get(category.upper(), ["Tidak ada rekomendasi untuk kategori ini."])


# ==============================
# RUN SERVER
# ==============================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
