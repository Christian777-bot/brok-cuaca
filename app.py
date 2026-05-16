from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "weather_xgboost.joblib"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"

st.set_page_config(
    page_title="Smart Weather Classifier",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(14,165,233,0.16), transparent 34%),
        radial-gradient(circle at top right, rgba(99,102,241,0.14), transparent 30%),
        linear-gradient(135deg, #f8fafc 0%, #eef6ff 45%, #ffffff 100%);
}

.block-container {
    padding-top: 2.1rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.10) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 14px !important;
}
section[data-testid="stSidebar"] label {
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] .stCaptionContainer {
    color: #cbd5e1 !important;
}

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 34px;
    padding: 34px 36px;
    margin-bottom: 22px;
    color: white;
    background:
        linear-gradient(135deg, rgba(2,132,199,0.96), rgba(79,70,229,0.92)),
        url('https://images.unsplash.com/photo-1504608524841-42fe6f032b4b?q=80&w=1600&auto=format&fit=crop');
    background-size: cover;
    background-position: center;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
}
.hero:after {
    content: "";
    position: absolute;
    right: -90px;
    top: -90px;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    background: rgba(255,255,255,0.16);
}
.hero-badge {
    display: inline-flex;
    gap: 8px;
    align-items: center;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.22);
    font-weight: 800;
    font-size: 13px;
    letter-spacing: .04em;
    text-transform: uppercase;
}
.hero-title {
    font-size: clamp(42px, 5vw, 72px);
    line-height: .98;
    font-weight: 950;
    margin: 18px 0 12px 0;
    letter-spacing: -0.05em;
}
.hero-subtitle {
    max-width: 760px;
    color: rgba(255,255,255,0.88);
    font-size: 18px;
    line-height: 1.65;
}

.card {
    background: rgba(255,255,255,0.86);
    border: 1px solid rgba(226,232,240,0.9);
    border-radius: 28px;
    padding: 26px;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    backdrop-filter: blur(10px);
}
.result-card {
    min-height: 330px;
}
.card-title {
    font-size: 14px;
    font-weight: 900;
    letter-spacing: .09em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 10px;
}
.weather-main {
    display: flex;
    align-items: center;
    gap: 18px;
    margin: 12px 0;
}
.weather-icon {
    width: 86px;
    height: 86px;
    border-radius: 28px;
    display: grid;
    place-items: center;
    font-size: 46px;
    background: linear-gradient(135deg, #e0f2fe, #eff6ff);
    border: 1px solid #dbeafe;
}
.weather-name {
    font-size: clamp(38px, 5vw, 58px);
    line-height: 1;
    font-weight: 950;
    color: #0f172a;
    letter-spacing: -0.04em;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 14px;
    border-radius: 999px;
    font-weight: 900;
    margin: 6px 0 15px 0;
}
.pill-high { background: #dcfce7; color: #166534; }
.pill-mid { background: #fef3c7; color: #92400e; }
.pill-low { background: #fee2e2; color: #991b1b; }
.desc {
    color: #334155;
    line-height: 1.72;
    font-size: 16px;
}
.status-box {
    margin-top: 16px;
    padding: 16px 18px;
    border-radius: 20px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #0f172a;
    font-weight: 800;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
}
.kpi {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 12px 28px rgba(15,23,42,0.05);
}
.kpi-label { color: #64748b; font-size: 13px; font-weight: 850; text-transform: uppercase; letter-spacing:.07em; }
.kpi-value { color: #0f172a; font-size: 32px; font-weight: 950; margin-top: 6px; }
.kpi-note { color: #64748b; font-size: 13px; line-height: 1.5; margin-top: 6px; }

.section-heading {
    font-size: 30px;
    font-weight: 950;
    color: #0f172a;
    letter-spacing: -0.03em;
    margin: 26px 0 12px 0;
}

.tip-card {
    height: 100%;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 14px 32px rgba(15,23,42,0.07);
}
.tip-icon { font-size: 26px; margin-bottom: 10px; }
.tip-title { font-size: 18px; font-weight: 950; color: #0f172a; margin-bottom: 8px; }
.tip-text { color: #475569; font-size: 15px; line-height: 1.65; }

.insight-row {
    display: grid;
    grid-template-columns: 34px 1fr;
    gap: 12px;
    align-items: start;
    padding: 13px 0;
    border-bottom: 1px solid #e2e8f0;
}
.insight-row:last-child { border-bottom: none; }
.insight-icon {
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    background: #e0f2fe;
}
.insight-title { font-weight: 900; color: #0f172a; margin-bottom: 3px; }
.insight-text { color: #64748b; line-height: 1.55; }

.footer-note {
    margin-top: 26px;
    padding: 18px 20px;
    border-radius: 22px;
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #78350f;
    line-height: 1.6;
}

button[kind="primary"] {
    border-radius: 16px !important;
    font-weight: 900 !important;
}

[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

WEATHER_GUIDE = {
    "Cloudy": {
        "id": "Berawan",
        "icon": "☁️",
        "status": "Cukup aman untuk aktivitas luar ruangan",
        "risk": "Risiko rendah sampai sedang",
        "summary": "Langit cenderung tertutup awan. Aktivitas luar ruangan masih memungkinkan, tetapi tetap siapkan antisipasi karena cuaca bisa berubah.",
        "tips": [
            ("☂️", "Bawa payung kecil", "Tidak wajib, tetapi berguna jika kondisi berubah menjadi gerimis."),
            ("🚶", "Aman untuk aktivitas ringan", "Cocok untuk pergi ke kampus, jalan santai, atau aktivitas luar singkat."),
            ("👕", "Pakai pakaian nyaman", "Suhu bisa terasa lembap, jadi gunakan pakaian yang tidak terlalu tebal."),
        ],
    },
    "Rainy": {
        "id": "Hujan",
        "icon": "🌧️",
        "status": "Perlu persiapan sebelum keluar rumah",
        "risk": "Risiko sedang sampai tinggi",
        "summary": "Kondisi mengarah ke hujan. Sebaiknya siapkan perlengkapan dan kurangi aktivitas luar ruangan yang tidak terlalu penting.",
        "tips": [
            ("☔", "Bawa payung atau jas hujan", "Ini perlengkapan utama agar aktivitas tetap aman dan nyaman."),
            ("🛵", "Hati-hati di jalan", "Jalan bisa licin dan jarak pandang dapat menurun."),
            ("💻", "Lindungi barang elektronik", "Masukkan laptop/HP ke tas yang aman dari air."),
        ],
    },
    "Sunny": {
        "id": "Cerah",
        "icon": "☀️",
        "status": "Baik untuk aktivitas luar ruangan",
        "risk": "Risiko panas/UV perlu diperhatikan",
        "summary": "Cuaca cenderung cerah. Ini bagus untuk aktivitas luar, namun tetap perhatikan paparan matahari dan kebutuhan cairan tubuh.",
        "tips": [
            ("🧴", "Gunakan pelindung matahari", "Topi atau sunscreen membantu mengurangi paparan UV."),
            ("💧", "Bawa air minum", "Tubuh lebih mudah dehidrasi saat cuaca panas."),
            ("🌳", "Cari tempat teduh", "Hindari terlalu lama di bawah matahari langsung."),
        ],
    },
    "Snowy": {
        "id": "Bersalju",
        "icon": "❄️",
        "status": "Butuh perlindungan dari suhu dingin",
        "risk": "Risiko dingin dan jalan licin",
        "summary": "Kondisi mengarah ke salju. Gunakan pakaian hangat dan lebih berhati-hati saat bepergian.",
        "tips": [
            ("🧥", "Gunakan jaket tebal", "Tubuh perlu perlindungan ekstra dari suhu dingin."),
            ("🥾", "Pakai alas kaki aman", "Permukaan jalan bisa licin saat bersalju."),
            ("🏠", "Kurangi aktivitas luar", "Keluar rumah seperlunya saja bila kondisi tidak mendukung."),
        ],
    },
}

COLUMN_LABELS = {
    "Temperature": "Suhu (°C)",
    "Humidity": "Kelembapan (%)",
    "Wind Speed": "Kecepatan Angin",
    "Precipitation (%)": "Peluang Hujan (%)",
    "Atmospheric Pressure": "Tekanan Udara",
    "UV Index": "Indeks UV",
    "Visibility (km)": "Jarak Pandang (km)",
    "Cloud Cover": "Kondisi Awan",
    "Season": "Musim",
    "Location": "Lokasi",
}

CATEGORY_LABELS = {
    "overcast": "Mendung tebal",
    "partly cloudy": "Sebagian berawan",
    "clear": "Cerah",
    "cloudy": "Berawan",
    "Spring": "Semi",
    "Summer": "Panas",
    "Autumn": "Gugur",
    "Winter": "Dingin",
    "inland": "Dataran dalam",
    "mountain": "Pegunungan",
    "coastal": "Pesisir",
}

WEATHER_ORDER = ["Cloudy", "Rainy", "Snowy", "Sunny"]
WEATHER_COLORS = {
    "Berawan": "#64748b",
    "Hujan": "#2563eb",
    "Bersalju": "#38bdf8",
    "Cerah": "#f59e0b",
}

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics_file():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}

@st.cache_data
def load_dataset_preview():
    data_path = ROOT_DIR / "weather_classification_data.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    return pd.DataFrame()


def display_name(value: str) -> str:
    return CATEGORY_LABELS.get(str(value), str(value).title())


def confidence_level(probability: float) -> tuple[str, str, str]:
    if probability >= 0.85:
        return "Keyakinan tinggi", "pill-high", "Prediksi sangat kuat"
    if probability >= 0.60:
        return "Keyakinan sedang", "pill-mid", "Masih ada kemungkinan cuaca lain"
    return "Keyakinan rendah", "pill-low", "Perlu cek peluang kelas lain"


def numeric_input(column: str, metadata: dict) -> float:
    values = metadata["numeric"][column]
    minimum = float(values["min"])
    maximum = float(values["max"])
    default = float(values["mean"])
    span = maximum - minimum
    step = 1.0 if span > 25 else 0.1
    return st.number_input(
        COLUMN_LABELS.get(column, column),
        min_value=minimum,
        max_value=maximum,
        value=round(default, 2),
        step=step,
        help=f"Rentang data training: {minimum:.2f} sampai {maximum:.2f}",
    )


def build_insights(user_input: dict) -> list[tuple[str, str, str]]:
    insights = []
    temp = float(user_input.get("Temperature", 0))
    humidity = float(user_input.get("Humidity", 0))
    wind = float(user_input.get("Wind Speed", 0))
    precipitation = float(user_input.get("Precipitation (%)", 0))
    uv = float(user_input.get("UV Index", 0))
    visibility = float(user_input.get("Visibility (km)", 0))
    cloud = str(user_input.get("Cloud Cover", ""))

    if precipitation >= 60:
        insights.append(("🌧️", "Peluang hujan tinggi", "Nilai precipitation cukup besar, jadi model wajar mempertimbangkan kemungkinan hujan."))
    elif precipitation >= 30:
        insights.append(("🌦️", "Peluang hujan sedang", "Ada sinyal hujan, tetapi belum terlalu dominan."))
    else:
        insights.append(("☀️", "Peluang hujan rendah", "Input precipitation rendah, sehingga kemungkinan cuaca kering lebih masuk akal."))

    if humidity >= 75:
        insights.append(("💧", "Kelembapan tinggi", "Udara terasa lebih lembap dan dapat mendukung kondisi berawan atau hujan."))
    elif humidity <= 35:
        insights.append(("🏜️", "Kelembapan rendah", "Udara cenderung kering sehingga kondisi cerah bisa lebih memungkinkan."))

    if uv >= 7:
        insights.append(("🧴", "Indeks UV tinggi", "Jika keluar rumah, sebaiknya gunakan pelindung dari sinar matahari."))
    elif uv <= 2:
        insights.append(("🌥️", "Indeks UV rendah", "Paparan matahari relatif rendah, sering terjadi saat cuaca berawan atau mendung."))

    if visibility <= 3:
        insights.append(("👀", "Jarak pandang rendah", "Perlu hati-hati saat berkendara karena pandangan bisa terbatas."))

    if wind >= 15:
        insights.append(("💨", "Angin cukup kencang", "Aktivitas luar ruangan perlu lebih berhati-hati, terutama membawa payung atau berkendara."))

    if cloud in ["overcast", "cloudy", "partly cloudy"]:
        insights.append(("☁️", "Kondisi awan terlihat jelas", f"Input kondisi awan adalah {display_name(cloud)}, sehingga prediksi berawan/hujan menjadi lebih masuk akal."))

    if temp >= 30:
        insights.append(("🔥", "Suhu cukup panas", "Perhatikan dehidrasi dan paparan panas saat beraktivitas di luar."))
    elif temp <= 5:
        insights.append(("❄️", "Suhu sangat dingin", "Perlu pakaian hangat dan perlindungan tambahan dari suhu rendah."))

    return insights[:5]


def make_probability_chart(probability_df: pd.DataFrame):
    fig = px.bar(
        probability_df.sort_values("Peluang (%)", ascending=True),
        x="Peluang (%)",
        y="Jenis Cuaca",
        orientation="h",
        text="Label Persen",
        color="Jenis Cuaca",
        color_discrete_map=WEATHER_COLORS,
        range_x=[0, 100],
    )
    fig.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
    fig.update_layout(
        height=370,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=70, t=20, b=20),
        xaxis_title="Peluang menurut model (%)",
        yaxis_title="",
        font=dict(size=14),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=False),
    )
    return fig


def render_tip_cards(tips):
    cols = st.columns(3)
    for col, (icon, title, text) in zip(cols, tips):
        with col:
            st.markdown(
                f"""
                <div class="tip-card">
                    <div class="tip-icon">{icon}</div>
                    <div class="tip-title">{title}</div>
                    <div class="tip-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_insights(insights):
    rows = ""
    for icon, title, text in insights:
        rows += f"""
        <div class="insight-row">
            <div class="insight-icon">{icon}</div>
            <div>
                <div class="insight-title">{title}</div>
                <div class="insight-text">{text}</div>
            </div>
        </div>
        """
    st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)


def main():
    if not MODEL_PATH.exists():
        st.error("Model belum tersedia. Pastikan file `models/weather_xgboost.joblib` sudah ada di GitHub, bukan di luar folder.")
        st.stop()

    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]
    metadata = artifact["feature_metadata"]
    metrics = artifact.get("metrics", {}) or load_metrics_file()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">🌦️ Machine Learning Weather App</div>
            <div class="hero-title">Smart Weather Classifier</div>
            <div class="hero-subtitle">
                Website prediksi tipe cuaca yang menerjemahkan hasil model menjadi bahasa sederhana:
                apa kondisi cuacanya, seberapa yakin modelnya, dan apa yang sebaiknya dilakukan pengguna.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("📌 Input Kondisi Cuaca")
        st.caption("Masukkan kondisi lingkungan. Nilai default memakai rata-rata dataset agar mudah dicoba.")
        st.divider()

        user_input = {}
        st.subheader("🌡️ Data Numerik")
        for column in metadata["numeric"]:
            user_input[column] = numeric_input(column, metadata)

        st.subheader("🏷️ Data Kategori")
        for column, options in metadata["categorical"].items():
            user_input[column] = st.selectbox(
                COLUMN_LABELS.get(column, column),
                options=options,
                format_func=display_name,
            )

        st.divider()
        st.caption("Tips: ubah nilai precipitation, humidity, cloud cover, dan UV index untuk melihat perubahan prediksi.")

    input_df = pd.DataFrame([user_input], columns=metadata["feature_columns"])
    encoded_prediction = int(pipeline.predict(input_df)[0])
    prediction = label_encoder.inverse_transform([encoded_prediction])[0]
    probabilities = pipeline.predict_proba(input_df)[0]

    probability_df = pd.DataFrame({"Weather Type": label_encoder.classes_, "Probability": probabilities})
    probability_df["Jenis Cuaca"] = probability_df["Weather Type"].map(lambda x: WEATHER_GUIDE.get(x, {}).get("id", x))
    probability_df["Peluang (%)"] = (probability_df["Probability"] * 100).round(2)
    probability_df["Label Persen"] = probability_df["Peluang (%)"].map(lambda x: f"{x:.1f}%")
    probability_df = probability_df.sort_values("Probability", ascending=False).reset_index(drop=True)

    top_probability = float(probability_df.iloc[0]["Probability"])
    second_probability = float(probability_df.iloc[1]["Probability"]) if len(probability_df) > 1 else 0.0
    guide = WEATHER_GUIDE.get(prediction, WEATHER_GUIDE["Cloudy"])
    confidence_text, confidence_class, confidence_note = confidence_level(top_probability)

    top_left, top_right = st.columns([1.45, 1], gap="large")

    with top_left:
        st.markdown(
            f"""
            <div class="card result-card">
                <div class="card-title">Hasil Prediksi Utama</div>
                <div class="weather-main">
                    <div class="weather-icon">{guide['icon']}</div>
                    <div>
                        <div class="weather-name">{guide['id']}</div>
                        <div class="pill {confidence_class}">● {top_probability * 100:.1f}% • {confidence_text}</div>
                    </div>
                </div>
                <div class="desc">{guide['summary']}</div>
                <div class="status-box">Status aktivitas: {guide['status']}<br>Level risiko: {guide['risk']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_right:
        accuracy = float(metrics.get("accuracy", 0))
        f1_macro = float(metrics.get("f1_macro", 0))
        precision = float(metrics.get("precision_macro", 0))
        recall = float(metrics.get("recall_macro", 0))
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Ringkasan Model</div>
                <div class="kpi-grid">
                    <div class="kpi"><div class="kpi-label">Akurasi</div><div class="kpi-value">{accuracy * 100:.1f}%</div><div class="kpi-note">Seberapa sering prediksi benar pada data uji.</div></div>
                    <div class="kpi"><div class="kpi-label">F1-Macro</div><div class="kpi-value">{f1_macro * 100:.1f}%</div><div class="kpi-note">Keseimbangan performa untuk semua kelas cuaca.</div></div>
                    <div class="kpi"><div class="kpi-label">Precision</div><div class="kpi-value">{precision * 100:.1f}%</div><div class="kpi-note">Ketepatan saat model memilih suatu kelas.</div></div>
                    <div class="kpi"><div class="kpi-label">Recall</div><div class="kpi-value">{recall * 100:.1f}%</div><div class="kpi-note">Kemampuan model menemukan kelas yang benar.</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-heading">✅ Rekomendasi Praktis untuk Pengguna</div>', unsafe_allow_html=True)
    render_tip_cards(guide["tips"])

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Peluang Cuaca", "🧠 Alasan Prediksi", "📈 Evaluasi Model", "📋 Data Input"])

    with tab1:
        col_chart, col_table = st.columns([1.45, .9], gap="large")
        with col_chart:
            st.markdown('<div class="section-heading">Peluang Setiap Jenis Cuaca</div>', unsafe_allow_html=True)
            st.caption("Angka ditampilkan dalam persen agar lebih mudah dipahami orang awam.")
            st.plotly_chart(make_probability_chart(probability_df), use_container_width=True)
        with col_table:
            st.markdown('<div class="section-heading">Interpretasi Singkat</div>', unsafe_allow_html=True)
            gap = (top_probability - second_probability) * 100
            if gap >= 25:
                message = "Prediksi utama cukup dominan dibanding kelas lain, sehingga hasilnya relatif kuat."
            elif gap >= 10:
                message = "Prediksi utama unggul, tetapi masih ada kelas lain yang perlu diperhatikan."
            else:
                message = "Peluang beberapa kelas cukup berdekatan, jadi hasil perlu dibaca lebih hati-hati."
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">Kesimpulan Probabilitas</div>
                    <div class="desc">Model paling yakin pada <b>{guide['id']}</b> dengan peluang <b>{top_probability*100:.1f}%</b>.</div>
                    <div class="desc" style="margin-top:12px;">Selisih dengan pilihan kedua: <b>{gap:.1f}%</b>.</div>
                    <div class="status-box">{message}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(probability_df[["Jenis Cuaca", "Peluang (%)"]], hide_index=True, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-heading">Kenapa Model Bisa Mengarah ke Hasil Ini?</div>', unsafe_allow_html=True)
        st.caption("Bagian ini memakai interpretasi sederhana dari input, supaya dosen dan pengguna awam bisa memahami konteksnya.")
        render_insights(build_insights(user_input))

    with tab3:
        st.markdown('<div class="section-heading">Evaluasi Model dalam Bahasa Sederhana</div>', unsafe_allow_html=True)
        train_rows = int(metrics.get("train_rows", 0))
        test_rows = int(metrics.get("test_rows", 0))
        model_name = metrics.get("model", "Machine Learning Model")
        st.markdown(
            f"""
            <div class="card">
                <div class="desc">
                    Model yang digunakan adalah <b>{model_name}</b>. Dataset dibagi menjadi data latih dan data uji.
                    Model belajar dari <b>{train_rows:,}</b> baris data latih dan diuji pada <b>{test_rows:,}</b> baris data uji.
                    Dengan akurasi sekitar <b>{accuracy*100:.1f}%</b>, model ini sudah cukup baik sebagai alat bantu prediksi awal.
                </div>
                <div class="footer-note">
                    Penting: hasil ini bukan pengganti data cuaca resmi real-time. Aplikasi ini cocok untuk demonstrasi machine learning,
                    edukasi, dan estimasi awal berdasarkan pola pada dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        report = metrics.get("classification_report", {})
        rows = []
        for cls in WEATHER_ORDER:
            if cls in report:
                rows.append({
                    "Jenis Cuaca": WEATHER_GUIDE.get(cls, {}).get("id", cls),
                    "Precision (%)": round(report[cls].get("precision", 0) * 100, 2),
                    "Recall (%)": round(report[cls].get("recall", 0) * 100, 2),
                    "F1-score (%)": round(report[cls].get("f1-score", 0) * 100, 2),
                    "Jumlah Data Uji": int(report[cls].get("support", 0)),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-heading">Data yang Dikirim ke Model</div>', unsafe_allow_html=True)
        readable_input = input_df.rename(columns=COLUMN_LABELS).copy()
        for col in readable_input.columns:
            if readable_input[col].dtype == object:
                readable_input[col] = readable_input[col].map(display_name)
        st.dataframe(readable_input, hide_index=True, use_container_width=True)

        data = load_dataset_preview()
        if not data.empty:
            with st.expander("Lihat contoh dataset yang digunakan"):
                st.dataframe(data.head(12), use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="footer-note">
            <b>Kesimpulan:</b> Website ini tidak hanya menampilkan output model, tetapi juga mengubahnya menjadi informasi yang bisa dipakai:
            prediksi cuaca, tingkat keyakinan, rekomendasi aktivitas, alasan sederhana, dan performa model.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
