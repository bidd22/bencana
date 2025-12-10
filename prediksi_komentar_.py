# prediksi_komentar_.py (versi final tanpa data_uji.csv)

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
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ----------------------------
# PATH DASAR FILES
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "D:\\bencana\\bencana\dataset_komentarbencana.csv")
TRAIN_CSV = os.path.join(BASE_DIR, "dataset_training.csv")
TEST_CSV = os.path.join(BASE_DIR, "dataset_testing.csv")
MODEL_FILE = os.path.join(BASE_DIR, "model_svm.pkl")

# ----------------------------
# INISIALISASI NLTK
# ----------------------------
try:
    nltk.download("stopwords", quiet=True)
except:
    pass

# ----------------------------
# STOPWORDS & NORMALISASI
# ----------------------------
try:
    stop_words = set(stopwords.words("indonesian"))
except:
    stop_words = set()

keep_words = {"tidak", "tapi", "namun", "bukan", "jangan", "belum", "kurang",
              "telah", "sudah", "bisa", "dapat", "akan", "sedang"}
stop_words = stop_words - keep_words

extra_stopwords = {
    "nih", "sih", "dong", "nya", "tuh", "deh", "kok", "aja", "saja",
    "yah", "wow", "oh", "hm", "hmm", "yup", "gan", "kak", "bro", "sis",
    "yuk", "loh", "kan", "yang", "dan", "di", "ke", "dari"
}
stop_words.update(extra_stopwords)

normalisasi_dict = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak",
    "klo": "kalau", "kalo": "kalau", "kl": "kalau",
    "krn": "karena", "tp": "tapi",
    "sy": "saya", "gw": "saya", "gue": "saya", "gua": "saya",
    "lu": "kamu", "lo": "kamu", "u": "kamu",
    "jd": "jadi", "sdh": "sudah", "udh": "sudah",
    "blm": "belum", "pls": "tolong", "tlg": "tolong",
    "gmna": "bagaimana", "gmn": "bagaimana",
    "bgt": "banget", "bngt": "banget"
}

# ----------------------------
# PREPROCESSING
# ----------------------------
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    clean_words = []
    for word in text.split():
        if word in normalisasi_dict:
            word = normalisasi_dict[word]
        if word not in stop_words:
            clean_words.append(word)
    return " ".join(clean_words)

# ----------------------------
# SAVE & LOAD MODEL
# ----------------------------
def save_model(model):
    joblib.dump(model, MODEL_FILE)
    print(f"Model disimpan di: {MODEL_FILE}")

def load_model():
    if os.path.exists(MODEL_FILE):
        print(f"Model dimuat dari: {MODEL_FILE}")
        return joblib.load(MODEL_FILE)
    print("Model belum ada, lakukan training dahulu.")
    return None

# ----------------------------
# SPLIT CSV SAVE
# ----------------------------
def save_split(df_train, df_test):
    df_train.to_csv(TRAIN_CSV, sep=";", index=False)
    df_test.to_csv(TEST_CSV, sep=";", index=False)
    print(f"dataset_training.csv & dataset_testing.csv berhasil dibuat!")

# ----------------------------
# TRAINING MODEL + SPLIT
# ----------------------------
def train_and_save_model():
    try:
        df = pd.read_csv(DATA_FILE, sep=";")
    except:
        print("Dataset utama tidak ditemukan.")
        return None

    if "full_text" not in df.columns or "labels" not in df.columns:
        print("Dataset harus punya kolom full_text dan labels.")
        return None

    df["clean_text"] = df["full_text"].apply(preprocess_text)

    X = df["clean_text"]
    y = df["labels"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    df_train = pd.DataFrame({"full_text": X_train, "labels": y_train})
    df_test = pd.DataFrame({"full_text": X_test, "labels": y_test})
    save_split(df_train, df_test)

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("svm", SVC(kernel="linear"))
    ])

    print("Melatih model...")
    model.fit(X_train, y_train)
    print("Training selesai.")

    preds = model.predict(X_test)
    print("\n--- Evaluasi Data Testing ---")
    print("Akurasi:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    save_model(model)
    return model

# ----------------------------
# KLASIFIKASI
# ----------------------------
def classify_new_comment(model, df_uji):
    if "full_text" not in df_uji.columns:
        print("CSV tidak memiliki kolom 'full_text'")
        return

    df_uji["clean_text"] = df_uji["full_text"].apply(preprocess_text)
    preds = model.predict(df_uji["clean_text"])

    label_map = {
        0: "Opini/Kritik",
        1: "Informasi",
        2: "Aksi/Bantuan"
    }

    df_uji["prediksi"] = [label_map[x] for x in preds]

    print("\n--- HASIL PREDIKSI ---")
    for i, row in df_uji.iterrows():
        print(f"{row['full_text']}\n→ {row['prediksi']}")
        print("-" * 30)

# ----------------------------
# MENU UTAMA
# ----------------------------
def main_menu():
    while True:
        print("\n==========================")
        print("   SISTEM KLASIFIKASI")
        print("==========================")
        print("1. Train Model + Split Dataset")
        print("2. Evaluasi Model (dataset_testing.csv)")
        print("3. Klasifikasi CSV (otomatis memakai dataset_testing.csv)")
        print("4. Klasifikasi Manual")
        print("5. Keluar")
        pilih = input("Pilih: ")

        if pilih == "1":
            train_and_save_model()

        elif pilih == "2":
            model = load_model()
            if model and os.path.exists(TEST_CSV):
                df_test = pd.read_csv(TEST_CSV, sep=";")
                classify_new_comment(model, df_test)
            else:
                print("Latih model terlebih dahulu.")

        elif pilih == "3":
            print("\nMemakai dataset_testing.csv...")
            model = load_model()
            if model and os.path.exists(TEST_CSV):
                df_uji = pd.read_csv(TEST_CSV, sep=";")
                classify_new_comment(model, df_uji)
            else:
                print("dataset_testing.csv belum ada. Lakukan training dahulu.")

        elif pilih == "4":
            model = load_model()
            if model:
                teks = input("Masukkan komentar: ")
                df_temp = pd.DataFrame({"full_text": [teks]})
                classify_new_comment(model, df_temp)

        elif pilih == "5":
            print("Keluar...")
            break

        else:
            print("Pilihan tidak valid.")

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    main_menu()
