import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Sistem Rekomendasi Minat Anak",
    page_icon="🎯",
    layout="wide"
)
st.markdown("""
<style>
    /* 1. Background Gradient & Smoothing */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a, #020617);
        color: #f8fafc;
    }

    /* 2. Judul Utama dengan Animasi Glowing */
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #60a5fa, #a855f7, #60a5fa);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 42px;
        font-weight: 900;
        margin-bottom: 5px;
        animation: shine 3s linear infinite;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }

    /* 3. Subtitle Center */
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 18px;
        margin-bottom: 40px;
    }

    /* 4. Glassmorphism Info Box */
    div[data-testid="stAlert"] {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        color: #cbd5e1 !important;
    }

    /* 5. Styling Radio Buttons (Opsi Jawaban) */
    div[role="radiogroup"] {
        gap: 10px !important;
    }

    div[role="radiogroup"] > label[data-baseweb="radio"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 12px 15px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    /* Hover effect */
    div[role="radiogroup"] > label[data-baseweb="radio"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: #60a5fa !important;
        transform: translateY(-2px);
    }

    /* Selected state with Glow */
    div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
        border: none !important;
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.4) !important;
    }

    div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* 6. Tombol Analisis "Neon" */
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
        color: white !important;
        border: none !important;
        padding: 15px 0px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        transition: 0.4s !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        box-shadow: 0 0 25px rgba(124, 58, 237, 0.6) !important;
        transform: scale(1.01);
    }

    /* 7. Label Pertanyaan */
    .stRadio > label p {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# KONFIGURASI DATASET
# =========================================================
CSV_PATH = "dataset_minat_anak.csv"
TARGET = "rekomendasi"

FEATURES = [
    "suka_main_game",
    "ingin_buat_game",
    "suka_coding",
    "suka_logika",
    "suka_matematika",
    "kreativitas_tinggi",
    "suka_merakit_lego",
    "suka_elektronik",
    "ingin_buat_robot",
    "suka_membongkar_barang"
]

# =========================================================
# LOAD DATASET
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()  # antisipasi spasi kolom
    return df

# =========================================================
# TRAINING & TESTING MODEL
# =========================================================
@st.cache_resource
def train_model(df):

    # ---------- X & y ----------
    X = df[FEATURES]                 # FITUR (INPUT)
    y = df[TARGET].astype(str)       # LABEL (OUTPUT)

    # ---------- SPLIT DATA ----------
    # 80% DATA LATIH (TRAINING)
    # 20% DATA UJI (TESTING)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,               # 20% testing
        random_state=42,
        stratify=y
    )

    # ---------- TRAINING MODEL ----------
    model = RandomForestClassifier(
        n_estimators=350,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # ---------- TESTING MODEL ----------
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    labels = model.classes_

    return model, acc, f1m, report, cm, labels

# =========================================================
# LOAD DATA & MODEL
# =========================================================
df = load_data()
model, acc, f1m, report, cm, labels = train_model(df)

# =========================================================
# HEADER
# =========================================================
st.title("🎯 Sistem Rekomendasi Minat Anak Berbasis AI")
st.write(f"Akurasi Model: **{acc:.2%}** | F1-Score: **{f1m:.2%}**")
st.divider()

# =========================================================
# INPUT KUISIONER
# =========================================================
st.subheader("Kuisioner Minat Anak")
st.info("Silakan pilih semua pertanyaan sebelum melakukan analisis.")

LEVEL_MAP = {
    "Tidak suka": 1,
    "Cukup suka": 2,
    "Suka": 4,
    "Sangat suka": 5
}

LABELS = {
    "suka_main_game": "Ketertarikan Bermain Game",
    "ingin_buat_game": "Minat Membuat Game",
    "suka_coding": "Antusiasme Belajar Coding",
    "suka_logika": "Kemampuan Logika",
    "suka_matematika": "Ketertarikan Matematika",
    "kreativitas_tinggi": "Tingkat Kreativitas",
    "suka_merakit_lego": "Hobi Merakit",
    "suka_elektronik": "Ketertarikan Elektronik",
    "ingin_buat_robot": "Minat Membuat Robot",
    "suka_membongkar_barang": "Eksperimen Bongkar Pasang"
}

inputs = {}
col1, col2 = st.columns(2)

for i, f in enumerate(FEATURES):
    with (col1 if i % 2 == 0 else col2):
        choice = st.radio(
    LABELS[f],
    options=list(LEVEL_MAP.keys()),
    index=None,
    horizontal=True,
    key=f
)
        inputs[f] = LEVEL_MAP.get(choice)

# =========================================================
# PREDIKSI
# =========================================================
if st.button("Analisis Rekomendasi", type="primary", use_container_width=True):

    # 1) Validasi: wajib semua dipilih
    belum = [LABELS[k] for k, v in inputs.items() if v is None]
    if belum:
        st.warning("Mohon pilih dahulu semua pertanyaan!")
        for b in belum:
            st.write(f"- {b}")
        st.stop()

    # =====================================================
    # RULE BASED (ATURAN TAMBAHAN) BIAR HASIL SESUAI LOGIKA
    # =====================================================

    # RULE 1: Jika semua jawab "Tidak suka" → jangan prediksi ke coding/robotic
    if all(v == 1 for v in inputs.values()):
        st.divider()
        st.warning("Hasil: Belum terlihat minat dominan karena semua jawaban adalah 'Tidak suka'.")
        st.info("Saran: coba ulangi penilaian setelah anak mencoba aktivitas sederhana (game edukasi, lego, coding dasar, dsb).")
        st.stop()

    # RULE 2: Jika semua jawab "Suka" / "Sangat suka" → hasil CAMPURAN
    if all(v >= 4 for v in inputs.values()):
        prediction = "CAMPURAN"
        probabilities = None
        classes = None
    else:
        # Data baru → SAMAKAN kolom dengan training
        x_new = pd.DataFrame([inputs]).reindex(columns=FEATURES)

        # Prediksi + probabilitas
        probabilities = model.predict_proba(x_new)[0]
        classes = model.classes_

        # Ambil 2 probabilitas tertinggi
        sorted_idx = probabilities.argsort()[::-1]
        top1, top2 = sorted_idx[0], sorted_idx[1]
        gap = probabilities[top1] - probabilities[top2]

        prediction = classes[top1]

        # RULE 3: Jika model "ragu" (selisih kecil) → CAMPURAN
        if gap < 0.10:   # kamu bisa ubah jadi 0.07 / 0.15 sesuai kebutuhan
            prediction = "CAMPURAN"

    # =====================================================
    # TAMPILKAN HASIL
    # =====================================================
    st.divider()

    st.markdown(f"""
    <div class="prediction-box">
        <p>HASIL ANALISIS</p>
        <h2>{prediction}</h2>
    </div>
    """, unsafe_allow_html=True)

    # Saran kurikulum berdasarkan hasil
    if prediction == "CODING":
        st.success("Prioritas: fokus pada logika perangkat lunak & pembuatan aplikasi/game.", icon="💻")
    elif prediction == "ROBOTIC":
        st.success("Prioritas: fokus pada perangkat keras, mekanik, sensor, dan robotika.", icon="🤖")
    else:
        st.success("Prioritas: fokus pada integrasi software + hardware secara seimbang (campuran).", icon="⚙️")

    # =====================================================
    # TAMPILKAN PROBABILITAS (HANYA JIKA ADA)
    # =====================================================
    if probabilities is not None:
        st.subheader("Skor Kecenderungan")
        for cls, prob in zip(classes, probabilities):
            st.write(cls)
            st.progress(float(prob))
