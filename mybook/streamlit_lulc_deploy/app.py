from pathlib import Path
import joblib
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# XGBoost/LightGBM dipakai sesuai notebook. Bila tidak tersedia, opsi tersebut dinonaktifkan.
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_model_lulc.joblib"

@st.cache_resource
def load_saved_model():
    return joblib.load(MODEL_PATH)
FEATURES = ["B02", "B03", "B04", "B08", "B11", "NDVI", "NDWI", "MNDWI", "NDBI"]
CLASS_NAMES = {
    0: "sawah", 1: "bangunan", 2: "danau",
    3: "lahan_hijau", 4: "mangrove", 5: "laut"
}
COLORS = {
    "sawah": "#F1C40F", "bangunan": "#E74C3C", "danau": "#3498DB",
    "lahan_hijau": "#27AE60", "mangrove": "#8E44AD", "laut": "#1F3A93"
}
DISPLAY_NAMES = {
    "sawah": "Sawah", "bangunan": "Bangunan", "danau": "Danau",
    "lahan_hijau": "Lahan Hijau", "mangrove": "Mangrove", "laut": "Laut"
}

st.set_page_config(page_title="WebGIS LULC Jawa Timur", page_icon="🛰️", layout="wide")
st.title("🛰️ Klasifikasi Tutupan Lahan (LULC) Jawa Timur")
st.caption("Sentinel-2A • Indeks spektral • Machine Learning • WebGIS interaktif")
st.info(
    "Aplikasi ini merupakan antarmuka Streamlit untuk proyek notebook. "
    "Untuk inferensi model, sediakan dataset berlabel atau file model yang telah dilatih. "
    "Peta titik contoh adalah titik representatif dari notebook, bukan peta raster klasifikasi penuh."
)

@st.cache_data
def load_sample_points():
    path = BASE_DIR / "sample_points.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

def add_indices(df):
    df = df.copy()
    eps = 1e-8
    required_bands = ["B02", "B03", "B04", "B08", "B11"]
    missing = [c for c in required_bands if c not in df.columns]
    if missing:
        raise ValueError("Kolom band tidak ditemukan: " + ", ".join(missing))
    for col in required_bands:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["NDVI"] = (df["B08"] - df["B04"]) / (df["B08"] + df["B04"] + eps)
    df["NDWI"] = (df["B03"] - df["B08"]) / (df["B03"] + df["B08"] + eps)
    df["MNDWI"] = (df["B03"] - df["B11"]) / (df["B03"] + df["B11"] + eps)
    df["NDBI"] = (df["B11"] - df["B08"]) / (df["B11"] + df["B08"] + eps)
    return df

def build_sample_map():
    points = load_sample_points()
    center = [-7.7, 112.5]
    m = folium.Map(location=center, zoom_start=7, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite", name="Google Satellite", overlay=False, control=True
    ).add_to(m)
    for class_key, samples in points.items():
        fg = folium.FeatureGroup(name=DISPLAY_NAMES.get(class_key, class_key), show=True)
        for p in samples:
            popup = (
                f"<b>{p.get('title', DISPLAY_NAMES.get(class_key, class_key))}</b><br>"
                f"Koordinat: {p.get('lat', 0):.5f}, {p.get('lon', 0):.5f}<br>"
                f"NDVI: {p.get('ndvi', float('nan')):.4f}<br>"
                f"NDWI: {p.get('ndwi', float('nan')):.4f}<br>"
                f"MNDWI: {p.get('mndwi', float('nan')):.4f}<br>"
                f"NDBI: {p.get('ndbi', float('nan')):.4f}"
            )
            folium.CircleMarker(
                location=[p["lat"], p["lon"]], radius=5,
                color="white", weight=1, fill=True, fill_opacity=0.9,
                fill_color=COLORS.get(class_key, "#777777"),
                tooltip=DISPLAY_NAMES.get(class_key, class_key),
                popup=folium.Popup(popup, max_width=280)
            ).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m

tab_map, tab_model, tab_about = st.tabs(["🗺️ WebGIS", "🧠 Klasifikasi Model", "ℹ️ Tentang"])

with tab_map:
    st.subheader("Peta WebGIS Tutupan Lahan Jawa Timur")
    html_path = BASE_DIR / "peta_lulc_jatim_webgis.html"
    raster_path = BASE_DIR / "peta_raster_lulc_jatim.png"
    if html_path.exists():
        st.success("Peta HTML dari notebook ditemukan.")
        st.components.v1.html(html_path.read_text(encoding="utf-8"), height=680, scrolling=True)
    else:
        st_folium(build_sample_map(), use_container_width=True, height=620, returned_objects=[])
        st.caption("Peta menampilkan titik sampel representatif per kelas dari notebook. Letakkan peta_lulc_jatim_webgis.html di folder aplikasi untuk menampilkan peta HTML hasil notebook.")
    if raster_path.exists():
        st.subheader("Raster hasil klasifikasi")
        st.image(Image.open(raster_path), use_container_width=True)
    else:
        st.warning("File peta_raster_lulc_jatim.png belum tersedia. Unggah/salin file ini dari lingkungan notebook jika ingin menampilkan raster.")

with tab_model:
    st.subheader("Latih model dari dataset berlabel atau lakukan prediksi")
    st.markdown(
        "Format dataset training mengikuti notebook: kolom band `B02`, `B03`, `B04`, `B08`, `B11` "
        "serta label `label` (angka 0–5) atau `label_teks`."
    )
    train_file = st.file_uploader("Unggah dataset training CSV", type=["csv"], key="train_csv")
    if train_file is not None:
        try:
            train_df = pd.read_csv(train_file)
            st.write("Pratinjau dataset", train_df.head())
            if "label" not in train_df.columns:
                if "label_teks" in train_df.columns:
                    reverse_map = {v: k for k, v in CLASS_NAMES.items()}
                    train_df["label"] = train_df["label_teks"].astype(str).str.lower().map(reverse_map)
                else:
                    st.error("Dataset harus memiliki kolom 'label' atau 'label_teks'.")
                    train_df = None
            if train_df is not None:
                train_df = add_indices(train_df)
                train_df = train_df.dropna(subset=FEATURES + ["label"])
                train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce")
                train_df = train_df.dropna(subset=["label"])
                train_df["label"] = train_df["label"].astype(int)
                choices = ["Random Forest"]
                if XGBClassifier is not None:
                    choices.append("XGBoost")
                if LGBMClassifier is not None:
                    choices.append("LightGBM")
                model_name = st.selectbox("Pilih algoritma", choices)
                if st.button("Latih model", type="primary"):
                    if train_df["label"].nunique() < 2:
                        st.error("Data harus memiliki setidaknya dua kelas label.")
                    else:
                        X = train_df[FEATURES]
                        y = train_df["label"]
                        stratify = y if y.value_counts().min() >= 2 else None
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=stratify
                        )
                        if model_name == "Random Forest":
                            model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
                        elif model_name == "XGBoost":
                            model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                                                  random_state=42, eval_metric="mlogloss")
                        else:
                            model = LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                                                   random_state=42, verbosity=-1)
                        with st.spinner("Melatih model..."):
                            model.fit(X_train, y_train)
                            pred = model.predict(X_test)
                        st.session_state["lulc_model"] = model
                        st.session_state["lulc_model_name"] = model_name
                        st.session_state["lulc_accuracy"] = accuracy_score(y_test, pred)
                        st.session_state["lulc_features"] = FEATURES
                        st.success(f"Model {model_name} selesai dilatih.")
                        st.metric("Akurasi test", f"{accuracy_score(y_test, pred):.2%}")
                        labels = sorted(set(y_test) | set(pred))
                        st.dataframe(pd.DataFrame(
                            confusion_matrix(y_test, pred, labels=labels),
                            index=[f"Aktual {CLASS_NAMES.get(int(x), x)}" for x in labels],
                            columns=[f"Prediksi {CLASS_NAMES.get(int(x), x)}" for x in labels]
                        ))
                        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
                        st.dataframe(pd.DataFrame(report).transpose())
        except Exception as e:
            st.error(f"Gagal memproses dataset: {e}")

    st.divider()
    st.subheader("Prediksi data baru dari CSV")
    st.write("CSV prediksi harus memiliki lima kolom band: B02, B03, B04, B08, B11.")
    pred_file = st.file_uploader("Unggah CSV untuk diprediksi", type=["csv"], key="predict_csv")
    if not MODEL_PATH.exists():
        st.error("File best_model_lulc.joblib tidak ditemukan.")
        st.stop()

    try:
        model = load_saved_model()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()
    if pred_file is not None:
        try:
            new_df = pd.read_csv(pred_file)
            prepared = add_indices(new_df)
            if model is None:
                st.error("Model LULC belum berhasil dimuat. Periksa file best_model_lulc.joblib.")
            else:
                valid = prepared.dropna(subset=FEATURES).copy()
                predictions = model.predict(valid[FEATURES])
                valid["prediksi_label"] = predictions
                valid["prediksi_kelas"] = [CLASS_NAMES.get(int(x), str(x)) for x in predictions]
                st.dataframe(valid.head(100), use_container_width=True)
                st.download_button(
                    "Unduh hasil prediksi CSV",
                    data=valid.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_lulc.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

with tab_about:
    st.subheader("Ringkasan metode dari notebook")
    st.markdown("""
    - **Wilayah studi:** Provinsi Jawa Timur.
    - **Sumber citra:** Sentinel-2 L2A melalui OpenEO.
    - **Band:** B02, B03, B04, B08, B11.
    - **Indeks spektral:** NDVI, NDWI, MNDWI, NDBI.
    - **Kelas:** sawah, bangunan, danau, lahan hijau, mangrove, laut.
    - **Algoritma yang dievaluasi dalam notebook:** Random Forest, XGBoost, dan LightGBM.
    - **Pembagian data pada notebook:** 80% training dan 20% testing; notebook juga membandingkan skor cross-validation 5-fold.

    Aplikasi ini tidak menjalankan proses pengambilan data OpenEO otomatis saat pengunjung membuka situs.
    Dataset, raster, dan HTML peta hasil notebook perlu disiapkan terlebih dahulu jika ingin ditampilkan sebagai hasil tetap.
    """)
