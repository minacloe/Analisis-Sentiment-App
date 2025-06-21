# Analisis Sentimen App (BiLSTM GUI)

Aplikasi **Analisis Sentimen Pelanggan Minimarket (Alfamart & Indomaret) di Media Sosial Twitter** berbasis antarmuka grafis (GUI) dengan model **Bidirectional LSTM (BiLSTM)**.  
Cocok untuk penelitian, pembelajaran machine learning, maupun demo analisis data berbasis AI.

---

## Fitur Utama

- 🔎 **Import Dataset CSV** dengan kolom `text` (ulasan) dan `sentiment` (label).
- 🚀 **Latih Model Deep Learning BiLSTM** langsung dari GUI (tanpa coding!).
- 📊 **Tampilan Data dan Hasil Prediksi** secara interaktif.
- ☁️ **Word Cloud**: Visualisasi kata-kata populer.
- 📜 **Classification Report**: Akurasi, Precision, Recall, F1-Score.
- 🔢 **Confusion Matrix**: Matriks perbandingan label asli vs prediksi.
- 📈 **Grafik Riwayat Training** (akurasi & loss tiap epoch).
- 💻 **Tanpa instalasi rumit**: Bisa dibuat file .exe dan langsung digunakan di Windows.

---

## Screenshot

<img src="https://github.com/minacloe/Analisis-Sentiment-App/raw/main/screenshot-app.png" alt="Tampilan Aplikasi" width="700"/>

---

## Cara Penggunaan

1. **Download / Clone repositori ini**
    ```bash
    git clone https://github.com/minacloe/Analisis-Sentiment-App.git
    cd Analisis-Sentiment-App
    ```

2. **Instal Library Python (minimal Python 3.8/3.11)**
    ```bash
    pip install -r requirements.txt
    ```
    **Library utama:**  
    - tensorflow, scikit-learn, pandas, numpy, matplotlib, seaborn, pillow, wordcloud

3. **Jalankan aplikasi**
    ```bash
    python appInteraktifAnalisisSentiment.py
    ```

4. **Import file CSV** yang berisi data ulasan, minimal kolom `text` dan `sentiment`.

5. **Klik “Latih Model BiLSTM”** → tunggu proses training selesai.

6. **Eksplorasi fitur visualisasi dan hasil prediksi** pada tab yang tersedia.

---

## Format Dataset Contoh

File CSV harus memuat kolom:
- `text` : Ulasan/kata-kata dari pelanggan.
- `sentiment` : Label sentimen (misal: positif, negatif, netral).

**Contoh:**
```csv
text,sentiment
"pelayanan kasir cepat dan ramah",positif
"parkir sempit dan antri lama",negatif
"barang lengkap dan harga murah",positif
...
