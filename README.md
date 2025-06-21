# Analisis Sentimen App (BiLSTM GUI - versi .exe)

Aplikasi Analisis Sentimen Pelanggan Minimarket (Alfamart & Indomaret) dari data Twitter, menggunakan Deep Learning **BiLSTM**, berbasis GUI (antarmuka visual) dan **langsung bisa dijalankan (.exe) tanpa instalasi Python**.

---
![Image](https://github.com/user-attachments/assets/a2f9e448-cf1b-46a0-b054-0fb4f72bc030)

## Fitur Aplikasi

- Import dataset CSV dari folder `Dataset/` (siap pakai).
- Latih model deep learning BiLSTM tanpa coding.
- Tampilkan hasil prediksi, metrik, dan visualisasi interaktif:
  - Word Cloud
   - ![image](https://github.com/user-attachments/assets/216a66b2-89e9-4440-a3b2-324d82d71888)

  - Pie Chart Sentiment
    - ![image](https://github.com/user-attachments/assets/3c933a00-259c-411c-a7cb-198442ab7e36)

  - Classification Report (Precision, Recall, F1, Akurasi)
    - ![image](https://github.com/user-attachments/assets/a7b48a19-4788-4023-a488-4610c28085a4)

  - Confusion Matrix
    - ![image](https://github.com/user-attachments/assets/67f51596-b9bb-4e8b-8cbb-f0f7dd4b4c69)

  - Grafik Riwayat Training
    - ![image](https://github.com/user-attachments/assets/3b0acd72-cb53-4af6-b7d7-294278f348f9)

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
   Double-click `appInteraktifAnalisisSentiment.exe`🖥️

4. **Pilih file CSV dataset**  
   Klik tombol **“Pilih File CSV”**, pilih dataset dari folder `/Dataset`.📑

   ![Screenshot 2025-06-22 012619](https://github.com/user-attachments/assets/d6867fa0-72fd-422a-aaa5-e75f81eaa4af)


6. **Klik “Latih Model BiLSTM”**  
   Tunggu proses training selesai (progres dapat dilihat di aplikasi).🕖
   ![Screenshot 2025-06-22 012649](https://github.com/user-attachments/assets/77adcdd0-1784-4662-9f7c-4bd3d5340996)
  ![image](https://github.com/user-attachments/assets/6867b219-5c8e-4462-a667-2b240429c6e1)



8. **Eksplorasi hasil**  
   Buka tab “Hasil Prediksi” dan gunakan tombol-tombol visualisasi yang tersedia.👁️‍🗨️
    ![image](https://github.com/user-attachments/assets/0b2936b8-3c1e-4fb7-b3d4-6f8a281c2283)

---

## Format Dataset

Pastikan file CSV memiliki **dua kolom** utama:  
- `text` : Teks/ulasan pelanggan  
- `sentiment` : Label sentimen (`positif`, `negatif`, `netral`)

**Contoh:**
```csv
text,sentiment
kasirnya ramah dan cepat,positif
antriannya lama, pelayanan kurang,negatif
...
