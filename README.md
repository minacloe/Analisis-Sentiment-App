# Analisis Sentimen App (BiLSTM GUI - versi .exe)

Aplikasi Analisis Sentimen Pelanggan Minimarket (Alfamart & Indomaret) dari data Twitter, menggunakan Deep Learning **BiLSTM**, berbasis GUI (antarmuka visual) dan **langsung bisa dijalankan (.exe) tanpa instalasi Python**.

---
![Image](https://github.com/user-attachments/assets/a2f9e448-cf1b-46a0-b054-0fb4f72bc030)

## Fitur Aplikasi

- Import dataset CSV dari folder `Dataset/` (siap pakai).
- Latih model deep learning BiLSTM tanpa coding.
- Tampilkan hasil prediksi, metrik, dan visualisasi interaktif:
  - Word Cloud
  - Classification Report (Precision, Recall, F1, Akurasi)
  - Confusion Matrix
  - Grafik Riwayat Training
- Friendly untuk peneliti, mahasiswa, dan siapa saja yang ingin eksplorasi analisis sentimen media sosial.

---

## Cara Menggunakan (Windows)

1. **Download file .exe aplikasi**
   - 🚀 Download Aplikasi EXE

[![Download Aplikasi Analisis Sentimen (.exe)](https://img.shields.io/badge/Download%20EXE-Analisis--Sentimen-blue?logo=windows&style=for-the-badge)](https://drive.google.com/uc?export=download&id=1i_uUr7ZvG6AFkPJu3k8LLv_nbPPVkwWD)

> **Klik tombol di atas untuk mengunduh aplikasi Analisis Sentimen versi .exe**  
> (Pastikan sudah login Google jika download lewat Google Drive!)

   - Tidak perlu install Python/lib tambahan.

2. **Download dataset contoh** dari folder [`/Dataset`](./Dataset):
   - File yang direkomendasikan:
     - `tweets_pelayanan_Alfamart_Cleansing_labeled.csv`
     - `tweets_pelayanan_Indomaret_Cleansing_labeled.csv`
     - `tweets_pelayanan_gabungan.csv`

3. **Jalankan aplikasi**  
   Double-click `appInteraktifAnalisisSentiment.exe`

4. **Pilih file CSV dataset**  
   Klik tombol **“Pilih File CSV”**, pilih dataset dari folder `/Dataset`.

5. **Klik “Latih Model BiLSTM”**  
   Tunggu proses training selesai (progres dapat dilihat di aplikasi).

6. **Eksplorasi hasil**  
   Buka tab “Hasil Prediksi” dan gunakan tombol-tombol visualisasi yang tersedia.

---

## Format Dataset

Pastikan file CSV memiliki **dua kolom** utama:  
- `text` : Teks/ulasan pelanggan  
- `sentiment` : Label sentimen (`positif`, `negatif`, `netral`)

**Contoh:**
```csv
text,sentiment
"kasirnya ramah dan cepat",positif
"antriannya lama, pelayanan kurang",negatif
...
