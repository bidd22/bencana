# Kelompok 6 - Mitigasi Bencana (A)
- Abid Arie F52124007
- Moh. Ikhsan F52124013
- Muh. Haikal F52124003
- Chindy Amelia Febriana F52124001
- Putri Ramadani Ma'ruf F52124018

# PROYEK:
    Analisis Sentimen Masyarakat Terhadap Kinerja Pemerintah Dalam Menangani Kasus Bencana Alam Hidrometereologi (Banjir Rob JABODETABEK) Dengan Memprediksi Komentar Bencana Alam Berdasarkan Dataset Komentar Twitter Masyarakat Untuk Diklasifikasi Kedalam Label Oleh Model ML - Algoritma SVM
    Proyek ini adalah implementasi Machine Learning untuk melakukan klasifikasi otomatis pada komentar atau teks yang berhubungan dengan kejadian bencana alam. Tujuannya adalah memilah teks menjadi kategori yang dapat ditindaklanjuti untuk mendukung manajemen informasi dan respon bencana.

# Fitur Utama Program
Program ini (`prediksi_komentar_.py`) mencakup langkah-langkah lengkap dari pembersihan data hingga prediksi model:
1.  **Preprocessing Khusus Bahasa Indonesia:** Melakukan *case folding*, penghapusan tautan, *mention*, tanda baca, dan menerapkan **Normalisasi Slang** (misalnya, `ga` menjadi `tidak`) sebelum *stopword removal*.
2.  **Visualisasi Data:** Menganalisis distribusi label, asal kota/kabupaten, dan karakteristik panjang teks.
3.  **Vektorisasi Fitur:** Menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)** untuk mengubah teks menjadi representasi numerik.
4.  **Model Klasifikasi:** Menggunakan algoritma **Support Vector Machine (SVM)** dengan *kernel linear* untuk pelatihan.
5.  **Evaluasi Model:** Melakukan *split* data dan menampilkan *Classification Report* dan Akurasi. Setelah training, kita pakai Scikit-learn Metrics buat ngukur seberapa pintar modelnya di Data Uji. Kita hitung Akurasi dan F1-Score sebagai bukti kalau model kita reliable.

Fungsi Prediksi Akhir: Program punya fungsi khusus (classify_new_comment) yang bisa ngambil input dari file CSV lain atau input manual. Fungsi ini langsung memproses dan ngasih prediksi labelnya (0, 1, atau 2).

Nilai Taktis: Prediksi inilah yang bikin sistem ini berguna. Kalau prediksi Informasi (1) tiba-tiba melonjak, itu sinyal darurat buat tim SAR harus ngecek lokasi. Kalau Kritik (0) yang melonjak, itu sinyal buat perbaikan statement atau pelayanan publik yang harus dilakukan segera demi kesejahteraan dan tingkat kepercayaan masyarakat terhadap instansi bisa baik

# Klasifikasi Label
Teks diklasifikasikan ke dalam tiga (3) kategori utama, sesuai dengan definisi dalam program:

| Label | Kategori | Deskripsi | Contoh Tujuan |
| :---: | :---: | :--- | :--- |
| **0** | Opini/Keluhan/Kritik | Komentar yang berisi pendapat, keluhan, atau kritik terhadap penanganan. | Perlu disaring untuk evaluasi publik. |
| **1** | Informasi | Komentar yang memberikan informasi faktual tentang kondisi, kerusakan, atau lokasi kejadian. | Sangat penting untuk pemetaan kondisi lapangan. |
| **2** | Aksi Pemerintah/Bantuan | Komentar yang berisi informasi tentang tindakan yang telah dilakukan (SAR, bantuan, evakuasi). | Digunakan untuk melacak kemajuan respon. |

# Persyaratan (Dependencies)
Untuk menjalankan program ini, Anda memerlukan lingkungan Python dan beberapa *library* berikut.

