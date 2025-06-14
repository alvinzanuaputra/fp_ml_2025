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

py -3.10 -m pip install tensorflow_hub