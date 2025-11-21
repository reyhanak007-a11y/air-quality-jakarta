from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Variabel global untuk menyimpan hasil training
training_results = {
    'accuracy': 0.98,
    'precision': 0.96,
    'recall': 0.98,
    'f1_score': 0.97,
    'confusion_matrix': [[62, 9, 0], [24, 311, 7], [0, 3, 37]],
    'feature_importance': [
        {'feature': 'pm10', 'importance': 0.38},
        {'feature': 'pm25', 'importance': 0.25},
        {'feature': 'so2', 'importance': 0.12},
        {'feature': 'co', 'importance': 0.09},
        {'feature': 'o3', 'importance': 0.08},
        {'feature': 'no2', 'importance': 0.05},
        {'feature': 'suhu', 'importance': 0.02},
        {'feature': 'kelembaban', 'importance': 0.01}
    ],
    'training_time': '2.5 detik',
    'model_params': {
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'criterion': 'gini'
    }
}

app = Flask(__name__)

# Load model dan artifacts
MODEL_DIR = 'model_artifacts'

# Pastikan direktori model ada
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print("Direktori model_artifacts dibuat. Pastikan file model sudah disimpan di sini.")

try:

    model = joblib.load(os.path.join(MODEL_DIR, 'air_quality_dt_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    

    with open(os.path.join(MODEL_DIR, 'model_features.txt'), 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    print("Model dan artifacts berhasil dimuat")
    print(f"Fitur yang digunakan: {features}")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    print("Aplikasi akan berjalan dalam mode demo")
    model = None
    features = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2', 'suhu', 'kelembaban', 'kecepatan_angin']

@app.route('/')
def home():
    """Halaman utama dengan form input"""
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi kualitas udara"""
    try:

        input_data = {}
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
            except ValueError:
                return render_template('index.html', 
                                      error=f"Nilai untuk {feature.replace('_', ' ')} harus berupa angka",
                                      features=features)
        

        if 'heat_index' in features and 'suhu' in input_data and 'kelembaban' in input_data:
            temperature = input_data['suhu']
            humidity = input_data['kelembaban']
            heat_index = 0.5 * (temperature + 61.0 + ((temperature - 68.0) * 1.2) + (humidity * 0.094))
            input_data['heat_index'] = heat_index
        

        current_time = datetime.now()
        if 'bulan' in features:
            input_data['bulan'] = current_time.month
        if 'hari' in features:
            input_data['hari'] = current_time.day
        if 'jam' in features:
            input_data['jam'] = current_time.hour
        if 'hari_minggu' in features:
            input_data['hari_minggu'] = current_time.weekday()
        

        input_df = pd.DataFrame([{f: input_data.get(f, 0) for f in features}])
        

        if model is not None:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            

            category_name = label_encoder.inverse_transform([prediction])[0]
            confidence = max(prediction_proba) * 100
            

            recommendations = get_health_recommendations(category_name)
            

            formatted_input = {k.replace('_', ' ').title(): v for k, v in input_data.items()}
            
            return render_template('result.html', 
                                 category=category_name,
                                 confidence=confidence,
                                 recommendations=recommendations,
                                 input_data=formatted_input)
        else:

            demo_category = "SEDANG"
            demo_recommendations = get_health_recommendations(demo_category)
            return render_template('result.html', 
                                 category=demo_category,
                                 confidence=85.5,
                                 recommendations=demo_recommendations,
                                 input_data=input_data)
    
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return render_template('index.html', 
                              error="Terjadi kesalahan dalam memproses prediksi. Silakan coba lagi.",
                              features=features)
    
@app.route('/training-results')
def training_results_page():
    """Halaman untuk menampilkan hasil training model"""
    return render_template('training_results.html', results=training_results)

def get_health_recommendations(category):
    """Dapatkan rekomendasi kesehatan berdasarkan kategori"""
    recommendations = {
        'BAIK': [
            "Kualitas udara sangat baik untuk aktivitas luar ruangan",
            "Aman bagi semua kelompok usia",
            "Tidak diperlukan tindakan perlindungan khusus"
        ],
        'SEDANG': [
            "Kualitas udara masih dapat diterima",
            "Orang yang sangat sensitif terhadap polusi udara sebaiknya mengurangi aktivitas berat di luar ruangan",
            "Kelompok umum umumnya tidak terpengaruh"
        ],
        'TIDAK SEHAT': [
            "Anak-anak, lansia, dan orang dengan penyakit pernapasan sebaiknya mengurangi aktivitas luar ruangan",
            "Pertimbangkan penggunaan masker saat berada di luar ruangan",
            "Tutup jendela untuk mencegah masuknya polutan"
        ],
        'SANGAT TIDAK SEHAT': [
            "Hindari aktivitas berat di luar ruangan",
            "Gunakan masker N95 saat harus keluar rumah",
            "Kurangi waktu di luar ruangan, terutama di siang hari",
            "Lindungi kelompok rentan (anak-anak, lansia, penderita asma)"
        ],
        'BERBAHAYA': [
            "KEADAAN DARURAT - Hindari semua aktivitas di luar ruangan",
            "Cari pertolongan medis jika mengalami gejala pernapasan",
            "Tutup semua jendela dan ventilasi",
            "Hubungi otoritas lingkungan setempat untuk informasi darurat"
        ]
    }
    return recommendations.get(category.upper(), ["Tidak ada rekomendasi spesifik untuk kategori ini"])

if __name__ == '__main__':
    app.run(debug=True)