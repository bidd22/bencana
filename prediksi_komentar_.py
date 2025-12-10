# prediksi_komentar_.py
import os
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # untuk menyimpan/men-load model
import sys

# ----------------------------
# Path dasar (simpan di folder yang sama dengan file .py)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'dataset_komentarbencana.csv')  # dataset asli
TRAIN_CSV = os.path.join(BASE_DIR, 'dataset_training.csv')
TEST_CSV = os.path.join(BASE_DIR, 'dataset_testing.csv')
MODEL_FILE = os.path.join(BASE_DIR, 'model_svm.pkl')

# ----------------------------
# Inisialisasi environment NLTK (jika perlu)
# ----------------------------
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception:
    pass

# ----------------------------
# Stopwords & normalisasi
# ----------------------------
try:
    stop_words = set(stopwords.words('indonesian'))
except Exception:
    # fallback: kecilkan risiko error bila stopwords tidak tersedia
    stop_words = set()

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

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_text(text):
    """Preprocessing teks: lower, remove mentions/url/digits/punct, normalisasi, hapus stopwords."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # ubah "heyyyy" -> "hey"
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

# ----------------------------
# Fungsi bantuan: simpan/load model & CSV split
# ----------------------------
def save_model(model, path=MODEL_FILE):
    joblib.dump(model, path)
    print(f"Model tersimpan di: {path}")

def load_model(path=MODEL_FILE):
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"Model berhasil dimuat dari: {path}")
        return model
    else:
        print("Model tidak ditemukan di path:", path)
        return None

def save_split_csv(df_train, df_test, train_path=TRAIN_CSV, test_path=TEST_CSV):
    df_train.to_csv(train_path, index=False, sep=';')
    df_test.to_csv(test_path, index=False, sep=';')
    print(f"Dataset training disimpan -> {train_path}")
    print(f"Dataset testing disimpan  -> {test_path}")

# ----------------------------
# Fungsi pelatihan lengkap (split + simpan + evaluasi)
# ----------------------------
def train_and_save_model(file_path, test_size=0.2, random_state=42, stratify=True, save_model_file=True):
    """Melatih model menggunakan SVM, menyimpan model dan dataset split, serta menampilkan evaluasi."""
    try:
        df = pd.read_csv(file_path, delimiter=';')
    except FileNotFoundError:
        print(f"File {file_path} tidak ditemukan.")
        return None, None, None
    except Exception as e:
        print("Error saat membaca file:", e)
        return None, None, None

    # Pastikan kolom penting ada
    if 'full_text' not in df.columns or 'labels' not in df.columns:
        print("Pastikan file memiliki kolom 'full_text' dan 'labels'.")
        return None, None, None

    # Preprocess
    df['clean_text'] = df['full_text'].apply(preprocess_text)
    X = df['clean_text']
    y = df['labels']

    stratify_param = y if stratify else None

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    # Simpan file CSV split (gabungkan kembali kolom original agar informatif)
    df_train = pd.DataFrame({'full_text': X_train, 'clean_text': X_train, 'labels': y_train})
    df_test = pd.DataFrame({'full_text': X_test, 'clean_text': X_test, 'labels': y_test})
    save_split_csv(df_train, df_test)

    # Siapkan pipeline
    text_classifier = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', SVC(kernel='linear', probability=False, random_state=42))
    ])

    # Latih dengan data training saja
    print("Mulai melatih model (hanya data training)...")
    text_classifier.fit(X_train, y_train)
    print("Pelatihan selesai.")

    # Evaluasi pada data test
    preds = text_classifier.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("\n--- Evaluasi pada Data Testing (20%) ---")
    print("Akurasi:", acc)
    print(classification_report(y_test, preds, zero_division=0))

    # Simpan model ke file
    if save_model_file:
        try:
            save_model(text_classifier, MODEL_FILE)
        except Exception as e:
            print("Gagal menyimpan model:", e)

    return text_classifier, (X_test, y_test), (X_train, y_train)

# ----------------------------
# Evaluasi model: confusion matrix + distribusi prediksi
# ----------------------------
def plot_confusion_matrix(y_true, y_pred, labels=[0,1,2], title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def plot_prediction_distribution(preds, title="Distribusi Prediksi"):
    plt.figure(figsize=(6,4))
    sns.countplot(x=preds)
    plt.title(title)
    plt.xlabel('Label Prediksi')
    plt.ylabel('Jumlah')
    plt.show()

# ----------------------------
# Fungsi klasifikasi (reuse dari kode awal, disesuaikan)
# ----------------------------
def classify_new_comment(model, input_data, source_type):
    """
    model: pipeline yang sudah fit
    input_data:
        - jika source_type == 'csv', input_data adalah DataFrame yang memiliki kolom 'full_text'
        - jika source_type == 'manual', input_data adalah list of strings
    """
    label_map = {
        0: "0 (Opini/Keluhan/Kritik)",
        1: "1 (Informasi)",
        2: "2 (Aksi Pemerintah/Bantuan)"
    }

    if source_type == 'csv':
        df_uji = input_data.copy()
        if 'full_text' not in df_uji.columns:
            print("CSV uji harus memiliki kolom 'full_text'.")
            return

        new_texts_raw = df_uji['full_text'].astype(str)
        new_texts_processed = new_texts_raw.apply(preprocess_text)
        predictions = model.predict(new_texts_processed)

        df_uji['predicted_label_index'] = predictions
        df_uji['predicted_category'] = df_uji['predicted_label_index'].map(label_map)

        print("\n--- Hasil Prediksi Komentar dari CSV ---")
        for index, row in df_uji.iterrows():
            print(f"[Komentar]: '{row['full_text'].strip()}'")
            print(f"[Prediksi]: {row['predicted_category']}")
            print("-" * 30)

        # Ringkasan statistik
        print("\n==============================================")
        print("📊 RINGKASAN HASIL KLASIFIKASI (TOTAL)")
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

        print("\n==============================================")
        print("📊 RINGKASAN HASIL KLASIFIKASI (MANUAL)")
        print("==============================================")
        df_manual = pd.DataFrame({'predicted_label_index': predictions})
        label_counts = df_manual['predicted_label_index'].map(label_map).value_counts().sort_index()
        total_baris = len(df_manual)
        print(f"Total Komentar Diuji: {total_baris}\n")
        for label, count in label_counts.items():
            percentage = (count / total_baris) * 100
            print(f"- {label}: {count} baris ({percentage:.1f}%)")
        print("==============================================")

# ----------------------------
# Mode input manual (CLI)
# ----------------------------
def manual_predict_mode(model):
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

# ----------------------------
# Mode Evaluate: gunakan model tersimpan + dataset_test.csv (yang dihasilkan saat split)
# ----------------------------
def evaluate_saved_model(model_path=MODEL_FILE, test_csv_path=TEST_CSV):
    model = load_model(model_path)
    if model is None:
        print("Tidak ada model untuk dievaluasi. Jalankan training dulu.")
        return
    if not os.path.exists(test_csv_path):
        print("Dataset test tidak ditemukan. Jalankan training dulu agar dataset_test disimpan.")
        return

    try:
        df_test = pd.read_csv(test_csv_path, delimiter=';')
    except Exception as e:
        print("Gagal memuat test CSV:", e)
        return

    if 'full_text' not in df_test.columns or 'labels' not in df_test.columns:
        print("File test CSV tidak memiliki kolom 'full_text' dan 'labels'.")
        return

    X_test = df_test['full_text'].astype(str).apply(preprocess_text)
    y_test = df_test['labels']
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("\n--- Evaluasi Model (dari file tersimpan) ---")
    print("Akurasi:", acc)
    print(classification_report(y_test, preds, zero_division=0))
    plot_confusion_matrix(y_test, preds, labels=sorted(y_test.unique()))
    plot_prediction_distribution(preds, title="Distribusi Prediksi pada Dataset Test")

# ----------------------------
# Fungsi utama: menu interaktif sederhana
# ----------------------------
def main_menu():
    print("\n==============================================")
    print("   SISTEM KLASIFIKASI KOMENTAR BENCANA")
    print("==============================================")
    print("Pilih opsi:")
    print("1. Train model (split 80/20, simpan model & dataset split)")
    print("2. Evaluate model (pakai model tersimpan + dataset_test.csv)")
    print("3. Klasifikasi dari CSV (pakai model tersimpan; file CSV harus punya kolom 'full_text')")
    print("4. Klasifikasi manual (input komentar satu-per-satu)")
    print("5. Load model dan langsung masuk manual predict mode")
    print("6. Keluar")
    choice = input(">> Pilih (1-6): ").strip()
    return choice

# ----------------------------
# Eksekusi utama
# ----------------------------
if __name__ == "__main__":
    while True:
        choice = main_menu()
        if choice == '1':
            # TRAIN
            print("Opsi: Train model dan simpan asset (dataset split + model).")
            model, test_set, train_set = train_and_save_model(DATA_FILE, test_size=0.2)
            if model is None:
                print("Pelatihan gagal. Pastikan dataset ada dan format sesuai (delimiter ';').")
        elif choice == '2':
            # EVALUATE
            print("Opsi: Evaluate model yang sudah tersimpan menggunakan dataset_testing.csv.")
            evaluate_saved_model()
        elif choice == '3':
            # KLASIFIKASI CSV
            model = load_model()
            if model is None:
                print("Model tidak ada. Silakan pilih opsi 1 untuk melatih model dulu.")
            else:
                csv_path = input("Masukkan path file CSV uji (default: data_uji.csv di folder yang sama). Tekan Enter untuk default: ").strip()
                if csv_path == '':
                    csv_path = os.path.join(BASE_DIR, 'data_uji.csv')
                if not os.path.exists(csv_path):
                    print("File CSV uji tidak ditemukan:", csv_path)
                else:
                    try:
                        df_uji = pd.read_csv(csv_path, delimiter=';')
                        classify_new_comment(model, df_uji, source_type='csv')
                    except Exception as e:
                        print("Gagal memproses CSV uji:", e)
        elif choice == '4':
            # MANUAL PREDICT (dengan model tersimpan)
            model = load_model()
            if model is None:
                print("Model tidak ada. Silakan pilih opsi 1 untuk melatih model dulu.")
            else:
                manual_predict_mode(model)
        elif choice == '5':
            # Load model and manual mode directly
            model = load_model()
            if model is None:
                print("Model tidak ada. Silakan pilih opsi 1 untuk melatih model dulu.")
            else:
                manual_predict_mode(model)
        elif choice == '6':
            print("Keluar. Sampai jumpa.")
            break
        else:
            print("Pilihan tidak dikenali. Silakan pilih 1-6.")
