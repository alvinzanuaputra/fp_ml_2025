
*Final Project Machine Learning*
Point :

1. LINK SVM CHRISTO (google colab) :
https://colab.research.google.com/drive/19a9fmC1j06coqnYpuSOqSC6V7BYg1gKi?usp=sharing

2. LINK SVM CHRISTO (github) : 
https://github.com/alvinzanuaputra/fp_ml_2025/blob/main/MachineLearning/IMDB_(Support_Vector_Machine).ipynb

3. LINK SUDAH SEMUA -ALVIN_NYOBA (google collab) :
https://drive.google.com/file/d/1EXXTFMD4tjvAxdnmwsiX9cxcYSpEN9di/view?usp=sharing

4. LINK LINK SUDAH SEMUA -ALVIN_NYOBA (github) : 
https://github.com/alvinzanuaputra/fp_ml_2025/blob/main/MachineLearning/main.ipynb

5. LINK POWER POINT (Rencana Awal)
https://www.canva.com/design/DAGoLz4SjyI/DYektuQqZsdGW0vohWWaIA/edit?utm_content=DAGoLz4SjyI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

6. LINK LAPORAN AKHIR :
https://docs.google.com/document/d/1Kps-0IWMAKSTloUYi7_0LhSyFMMVzmFf/edit?usp=sharing&ouid=103471198710808240156&rtpof=true&sd=true



```
1. SVM untuk IMDB: Baseline Cepat & Andal
Hubungan: Teks ulasan film di IMDB seringkali cukup panjang dan kaya kosakata. Representasinya menggunakan TF-IDF, yang menghasilkan data berdimensi tinggi dan sparse.
SVM sangat cocok untuk jenis data seperti ini karena:
Mampu menangani ratusan ribu fitur (kata)
Cepat dilatih
Jarang overfitting
Kapan digunakan: Ideal saat Anda butuh baseline cepat dan efisien untuk membandingkan model lain.
Cocok untuk: Skenario baseline atau sistem real-time sederhana

2. ANN untuk IMDB: Menangkap Pola Bahasa Kompleks
Hubungan: Bahasa dalam review film bisa ambigu dan non-linear, seperti:
“The story was slow and boring, but the acting saved it.”
ANN mampu:
Menangkap pola seperti ironi, kontras, atau struktur kalimat rumit
Belajar interaksi antar kata secara non-linear
Meskipun ANN sederhana seperti dalam proyek Anda hanya pakai TF-IDF sebagai input, ia tetap bisa belajar pola yang tidak bisa ditangkap SVM.
 Cocok untuk: Mendeteksi sentimen kompleks, kalimat ambigu, atau saat kualitas lebih diutamakan dari kecepatan

3. XGBoost untuk IMDB: Akurasi Tinggi + Kontrol Regularisasi
Hubungan: Review IMDB tidak selalu konsisten — kadang sangat positif, kadang sangat negatif, kadang campuran. Model yang fleksibel seperti XGBoost mampu:
Belajar pola dari review-review ekstrem atau ambigu
Mencegah overfitting walau datanya besar
Menyediakan performa top-tier secara umum
XGBoost sering menjadi juara di kompetisi NLP berbasis fitur manual (seperti TF-IDF), karena kekuatannya dalam menangani noise, outlier, dan fitur penting otomatis.
 Cocok untuk: Proyek produksi atau saat akurasi tertinggi sangat dibutuhkan.

```

```bash
py -3.10 -m pip install tensorflow_hub
```