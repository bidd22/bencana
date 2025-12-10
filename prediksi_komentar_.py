import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os # Import pustaka os untuk cek file

# 1. Load Dataset dan Visualisasi Awal
# Path file data training Anda
data_file = 'dataset_komentarbencana.csv'
df = None # Definisikan df di luar try/except

try:
    df = pd.read_csv(data_file, sep=';') # Menggunakan delimiter titik koma sesuai dataset Anda
    print("Data training berhasil dimuat!")
    print(df.info())

    # --- 1. Visualisasi Distribusi Label (Target) ---
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='labels', data=df, palette='viridis', hue='labels', legend=False)
    plt.title('Distribusi Kelas (Labels)', fontsize=15)
    plt.xlabel('Label (0=Opini, 1=Informasi, 2=Aksi)', fontsize=12)
    plt.ylabel('Jumlah Data', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.show()

    # --- 2. Visualisasi Distribusi Kota/Kabupaten ---
    plt.figure(figsize=(10, 6))
    top_cities = df['kota_kabupaten'].value_counts().nlargest(10).index
    sns.countplot(y='kota_kabupaten', data=df[df['kota_kabupaten'].isin(top_cities)],
                  order=top_cities, palette='magma', hue='kota_kabupaten', legend=False)
    plt.title('Top 10 Asal Kota/Kabupaten', fontsize=15)
    plt.xlabel('Jumlah', fontsize=12)
    plt.ylabel('Kota/Kabupaten', fontsize=12)
    plt.show()

    # --- 3. Analisis Karakteristik Teks ---
    df['text_length'] = df['full_text'].astype(str).apply(len)
    df['word_count'] = df['full_text'].astype(str).apply(lambda x: len(x.split()))

    # Plot Distribusi Panjang Teks (Karakter)
    plt.figure(figsize=(10, 5))
    sns.histplot(df['text_length'], bins=30, kde=True, color='skyblue')
    plt.title('Distribusi Panjang Teks (Jumlah Karakter)', fontsize=15)
    plt.xlabel('Panjang Teks', fontsize=12)
    plt.ylabel('Frekuensi', fontsize=12)
    plt.show()

    # Plot Distribusi Jumlah Kata
    plt.figure(figsize=(10, 5))
    sns.histplot(df['word_count'], bins=30, kde=True, color='salmon')
    plt.title('Distribusi Jumlah Kata per Komentar', fontsize=15)
    plt.xlabel('Jumlah Kata', fontsize=12)
    plt.ylabel('Frekuensi', fontsize=12)
    plt.show()

except FileNotFoundError:
    print(f"File {data_file} tidak ditemukan. Model tidak dapat dilatih.")
except Exception as e:
    print(f"Error saat memuat atau memvisualisasikan data training: {e}")


# ==========================================
# 2. PREPROCESSING DAN UTILITIES
# ==========================================

# Download resource NLTK jika belum ada
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# --- Setup Stopwords ---
stop_words = set(stopwords.words('indonesian'))
keep_words = {
    "tidak", "tapi", "namun", "bukan", "jangan", "belum", "kurang",
    "telah", "sudah", "bisa", "dapat", "akan", "sedang"
}
stop_words = stop_words - keep_words
extra_stopwords = {
    "nih", "sih", "dong", "nya", "tuh", "deh", "kok", "aja", "saja",
    "yah", "wow", "oh", "hm", "hmm", "yup", "gan", "kak", "bro", "sis",
    "yuk", "loh", "kan", "yang", "dan", "di", "ke", "dari"
}
stop_words.update(extra_stopwords)

# --- Kamus Normalisasi (Pengganti Stemming) ---
normalisasi_dict = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak", "tdk": "tidak", "tak": "tidak",
    "klo": "kalau", "kalo": "kalau", "kl": "kalau", "dg": "dengan", "dgn": "dengan", "yg": "yang",
    "krn": "karena", "karna": "karena", "tp": "tapi", "sm": "sama", "utk": "untuk", "dlm": "dalam",
    "dr": "dari", "sy": "saya", "gw": "saya", "aku": "saya", "gue": "saya", "gua": "saya",
    "lu": "kamu", "lo": "kamu", "u": "kamu", "jd": "jadi", "jdi": "jadi", "sdh": "sudah",
    "udh": "sudah", "dah": "sudah", "blm": "belum", "tlg": "tolong", "pls": "tolong", 
    "mohon": "tolong", "bntu": "bantu", "knp": "kenapa", "napa": "kenapa", "gmn": "bagaimana", 
    "gmna": "bagaimana", "bgt": "banget", "bngit": "banget", "bngt": "banget", "bkn": "bukan",
    "org": "orang", "dpt": "dapat", "lbh": "lebih", "bnjr": "banjir", "evaku": "evakuasi",
    "korbn": "korban", "tnda": "tenda", "bntuan": "bantuan"
}

def preprocess_text(text):
    """Preprocessing Teks."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    clean_words = []
    for word in words:
        if word in normalisasi_dict:
            word = normalisasi_dict[word]
        if word not in stop_words:
            clean_words.append(word)
    return " ".join(clean_words)


# ==========================================
# 3. FUNGSI PELATIHAN MODEL
# ==========================================

def train_and_save_model(file_path):
    """Melatih model klasifikasi menggunakan SVM."""

    try:
        df = pd.read_csv(file_path, delimiter=';')
    except Exception as e:
        print(f"Error saat memuat file training: {e}")
        return None 

    # Memastikan kolom yang dibutuhkan ada
    if 'full_text' not in df.columns or 'labels' not in df.columns:
        print("Pastikan file memiliki kolom 'full_text' dan 'labels'.")
        return None 

    df['clean_text'] = df['full_text'].apply(preprocess_text)
    X = df['clean_text']
    y = df['labels']

    text_classifier = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', SVC(kernel='linear', random_state=42))
    ])

    print("Mulai melatih model SVM...")
    text_classifier.fit(X, y)
    print("Pelatihan model selesai.")

    # Evaluasi model (dibuat ringkas)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    text_classifier.fit(X_train, y_train)
    predictions = text_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print("\n--- Akurasi Model: ", accuracy)
    print("---\n--- Laporan Klasifikasi (di Data Uji) ---")
    print(classification_report(y_test, predictions, target_names=['0 (Opini/Kritik)', '1 (Informasi)', '2 (Aksi/Bantuan)'], zero_division=0))

    text_classifier.fit(X, y)
    return text_classifier


# ==========================================
# 4. FUNGSI KLASIFIKASI DAN PENGHITUNGAN LABEL
# ==========================================

def classify_new_comment(model, input_data, source_type):
    """
    Menggunakan model untuk memprediksi label komentar, baik dari CSV atau input manual.
    """

    # Mapping Label
    label_map = {
        0: "0 (Opini/Keluhan/Kritik)",
        1: "1 (Informasi)",
        2: "2 (Aksi Pemerintah/Bantuan)"
    }
    
    if source_type == 'csv':
        df_uji = input_data
        new_texts_raw = df_uji['full_text'].astype(str)
        new_texts_processed = new_texts_raw.apply(preprocess_text)
        
        # 2. Lakukan Prediksi
        predictions = model.predict(new_texts_processed)
        
        # 3. Gabungkan hasil prediksi ke DataFrame uji
        df_uji['predicted_label_index'] = predictions
        df_uji['predicted_category'] = df_uji['predicted_label_index'].map(label_map)

        # 4. Tampilkan Hasil Prediksi per Baris
        print("\n--- Hasil Prediksi Komentar Terbaru dari CSV ---")
        for index, row in df_uji.iterrows():
            print(f"[Komentar]: '{row['full_text'].strip()}'")
            print(f"[Prediksi]: {row['predicted_category']}")
            print("-" * 30)

        # 5. HITUNG DAN TAMPILKAN RINGKASAN STATISTIK
        print("\n==============================================")
        print("RINGKASAN HASIL KLASIFIKASI (TOTAL)")
        print("==============================================")
        
        label_counts = df_uji['predicted_category'].value_counts().sort_index()
        total_baris = len(df_uji)
        print(f"Total Komentar Diuji: {total_baris}\n")
        
        for label, count in label_counts.items():
            percentage = (count / total_baris) * 100
            print(f"- {label}: {count} baris ({percentage:.1f}%)")

        print("==============================================")
    
    elif source_type == 'manual':
        comments = input_data
        if not comments:
            return

        new_texts_processed = [preprocess_text(c) for c in comments]
        predictions = model.predict(new_texts_processed)

        print("\n--- Hasil Prediksi Komentar Manual ---")
        for text_raw, prediction_index in zip(comments, predictions):
            predicted_category = label_map.get(prediction_index)
            print(f"[Komentar]: '{text_raw.strip()}'")
            print(f"[Prediksi]: {predicted_category}")
            print("-" * 30)
            
        # Tampilkan ringkasan statistik untuk input manual
        print("\n==============================================")
        print("RINGKASAN HASIL KLASIFIKASI (MANUAL)")
        print("==============================================")
        
        df_manual = pd.DataFrame({'predicted_label_index': predictions})
        label_counts = df_manual['predicted_label_index'].map(label_map).value_counts().sort_index()
        total_baris = len(df_manual)
        print(f"Total Komentar Diuji: {total_baris}\n")
        
        for label, count in label_counts.items():
            percentage = (count / total_baris) * 100
            print(f"- {label}: {count} baris ({percentage:.1f}%)")

        print("==============================================")


def manual_predict_mode(model):
    """Meminta input komentar manual dari pengguna secara berulang."""
    print("\n========================================================")
    print("      MODE UJI COBA MANUAL: Input Komentar")
    print("========================================================")
    print("Masukkan komentar (atau ketik 'selesai' untuk mengakhiri):")
    
    comments_to_predict = []
    
    while True:
        try:
            comment = input(">> Komentar: ")
            if comment.lower() == 'selesai':
                break
            if comment.strip():
                comments_to_predict.append(comment.strip())
        except EOFError:
            break
        except KeyboardInterrupt:
            break

    if comments_to_predict:
        classify_new_comment(model, comments_to_predict, source_type='manual')
    else:
        print("Tidak ada komentar yang dimasukkan.")


# ==========================================
# 5. BAGIAN UTAMA PROGRAM (MAIN EXECUTION BLOCK)
# ==========================================

if __name__ == "__main__":

    # 1. Melatih model
    classifier_model = train_and_save_model(data_file)
    
    if classifier_model is None:
        print("\nModel gagal dilatih. Program dihentikan.")
    else:
        # 2. Path file data uji Anda
        uji_file = 'data_uji.csv' 
        
        # 3. Cek keberadaan file uji dan lakukan klasifikasi
        if os.path.exists(uji_file):
            print(f"\nFile Uji '{uji_file}' ditemukan. Memulai klasifikasi menggunakan CSV...")
            
            try:
                df_uji = pd.read_csv(uji_file, delimiter=';')
                classify_new_comment(classifier_model, df_uji, source_type='csv')
            except Exception as e:
                print(f"Error saat memproses file CSV uji: {e}")
                print("\nGagal memproses CSV. Beralih ke mode input manual.")
                manual_predict_mode(classifier_model)
        else:
            print(f"\nFile Uji '{uji_file}' tidak ditemukan.")
            manual_predict_mode(classifier_model)