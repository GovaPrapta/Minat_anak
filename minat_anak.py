import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Interest Discovery System",
    page_icon="🎯", 
    layout="wide"
)
st.markdown("""
<style>
/* Background app */
.main {
    background-color: #0b1220; /* aman untuk dark theme */
}

/* === LABEL PERTANYAAN (AGAR KELIHATAN) === */
.stRadio [data-testid="stMarkdownContainer"] p,
.stRadio label p{
    font-weight: 800 !important;
    color: #f9fafb !important;              /* putih */
    font-size: 16px !important;
    margin-bottom: 10px !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.55) !important;
}

/* Container radio */
div[role="radiogroup"]{
    gap: 10px !important;
}

/* Item radio */
div[role="radiogroup"] > label[data-baseweb="radio"]{
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 10px 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.10);
    transition: all .2s ease;
    cursor: pointer;
    min-height: 44px;
    display: flex;
    align-items: center;
}

/* Hover */
div[role="radiogroup"] > label[data-baseweb="radio"]:hover{
    border-color: #3b82f6;
    box-shadow: 0 6px 18px rgba(59,130,246,0.25);
    transform: translateY(-1px);
}

/* Hilangkan bulatan radio default */
div[role="radiogroup"] > label[data-baseweb="radio"] span:first-child{
    display: none !important;
}

/* Teks opsi radio */
div[role="radiogroup"] > label[data-baseweb="radio"] span{
    font-weight: 800;
    color: #111827;
    font-size: 14px;
}

/* 🔵 SAAT DIPILIH → BIRU SOLID */
div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked){
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    border: 1px solid #1d4ed8;
    box-shadow: 0 10px 26px rgba(37,99,235,0.50);
}

/* Teks jadi putih saat dipilih */
div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) span{
    color: #ffffff;
}

/* Spasi antar pertanyaan */
div[data-testid="column"] .stRadio{
    margin-bottom: 14px;
}

/* Box hasil prediksi */
.prediction-box {
    padding: 20px;
    border-radius: 14px;
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 22px rgba(0,0,0,0.10);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


CSV_PATH = "dataset_minat_anak.csv"
TARGET = "rekomendasi"
FEATURES = [
    "suka_main_game", "ingin_buat_game", "suka_coding", "suka_logika",
    "suka_matematika", "kreativitas_tinggi", "suka_merakit_lego",
    "suka_elektronik", "ingin_buat_robot", "suka_membongkar_barang"
]

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df[TARGET].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=350, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")
    rep = classification_report(y_test, pred, digits=4)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    return model, acc, f1m, rep, cm, labels

# --- LOAD DATA & MODEL ---
try:
    df = load_data()
    model, acc, f1m, rep, cm, labels = train_model(df)
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# --- HEADER SECTION ---
col_logo, col_title = st.columns([1, 4])

with col_title:
    st.title("Interest Discovery System")
    st.caption("Sistem Rekomendasi Program Pembelajaran Berbasis AI")

st.divider()

# --- INPUT SECTION ---
st.subheader("Kuisioner Minat")
st.info("Silakan pilih tingkat ketertarikan anak pada setiap kategori di bawah ini.", icon=":material/info:")

LEVEL_MAP = {"Tidak suka": 1, "Cukup suka": 2, "Suka": 4, "Sangat suka": 5}
LABELS = {
    "suka_main_game": "Ketertarikan Bermain Game",
    "ingin_buat_game": "Minat Membuat Game",
    "suka_coding": "Antusiasme Belajar Coding",
    "suka_logika": "Kemampuan Logika & Teka-teki",
    "suka_matematika": "Ketertarikan pada Matematika",
    "kreativitas_tinggi": "Tingkat Kreativitas",
    "suka_merakit_lego": "Hobi Merakit (Lego/Puzzle)",
    "suka_elektronik": "Rasa Ingin Tahu Alat Elektronik",
    "ingin_buat_robot": "Keinginan Merancang Robot",
    "suka_membongkar_barang": "Eksperimen Bongkar Pasang Barang",
}

inputs = {}
col1, col2 = st.columns(2, gap="large")

for i, f in enumerate(FEATURES):
    label = LABELS.get(f, f.replace("_", " ").title())
    with col1 if i % 2 == 0 else col2:
        # Menggunakan Radio dengan layout horizontal yang bersih
        choice = st.radio(
            label,
            options=list(LEVEL_MAP.keys()),
            horizontal=True,
            key=f,
            help=f"Berikan penilaian untuk {label}"
        )
        inputs[f] = LEVEL_MAP[choice]

st.markdown("<br>", unsafe_allow_html=True)

# --- PREDICTION LOGIC ---
if st.button("Analisis Rekomendasi", type="primary", use_container_width=True):
    x_new = pd.DataFrame([inputs])
    prediction = model.predict(x_new)[0]
    probabilities = model.predict_proba(x_new)[0]
    classes = model.classes_
    
    st.divider()
    
    # Hasil dalam Kolom
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.markdown(f"""
            <div class="prediction-box">
                <p style="margin-bottom:0; color:#666;">HASIL ANALISIS</p>
                <h2 style="color:#007BFF; margin-top:0;">{prediction}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # hasil rekomnedasi
        if prediction == "CODING":
            st.success("Prioritas Kurikulum: Fokus pada pengembangan logika perangkat lunak dan pembuatan aplikasi/game.", icon=":material/code:")
        elif prediction == "ROBOTIC":
            st.success("Prioritas Kurikulum: Fokus pada perangkat keras, perakitan mekanik, dan pemrograman sensor.", icon=":material/precision_manufacturing:")
        else:
            st.success("Prioritas Kurikulum: Fokus pada integrasi antara perangkat keras dan perangkat lunak secara seimbang.", icon=":material/settings_input_component:")

    with res_col2:
        st.write("**Skor Kecenderungan:**")
        for cls, prob in zip(classes, probabilities):
            st.write(f"{cls}")
            st.progress(prob)
            
    with st.expander("Data Teknis Analisis"):
        prob_df = pd.DataFrame({"Kategori": classes, "Probabilitas": probabilities}).sort_values("Probabilitas", ascending=False)
        st.table(prob_df)