from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model_lulc.joblib"
FEATURES = ["B02", "B03", "B04", "B08", "B11", "NDVI", "NDWI", "MNDWI", "NDBI"]
CLASS_NAMES = {
    0: "sawah", 1: "bangunan", 2: "danau",
    3: "lahan_hijau", 4: "mangrove", 5: "laut",
}
DISPLAY_NAMES = {
    "sawah": "Sawah", "bangunan": "Bangunan", "danau": "Danau",
    "lahan_hijau": "Lahan Hijau", "mangrove": "Mangrove", "laut": "Laut",
}
COLORS = {
    "sawah": "#F1C40F", "bangunan": "#E74C3C", "danau": "#3498DB",
    "lahan_hijau": "#27AE60", "mangrove": "#8E44AD", "laut": "#1F3A93",
}

st.set_page_config(page_title="WebGIS LULC Jawa Timur", page_icon="🛰️", layout="wide")

@st.cache_resource
def load_saved_model():
    """Load model once and reuse it across Streamlit reruns."""
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_sample_points():
    path = BASE_DIR / "sample_points.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def add_indices(df):
    df = df.copy()
    eps = 1e-8
    bands = ["B02", "B03", "B04", "B08", "B11"]
    missing = [c for c in bands if c not in df.columns]
    if missing:
        raise ValueError("Kolom band tidak ditemukan: " + ", ".join(missing))
    for col in bands:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["NDVI"] = (df["B08"] - df["B04"]) / (df["B08"] + df["B04"] + eps)
    df["NDWI"] = (df["B03"] - df["B08"]) / (df["B03"] + df["B08"] + eps)
    df["MNDWI"] = (df["B03"] - df["B11"]) / (df["B03"] + df["B11"] + eps)
    df["NDBI"] = (df["B11"] - df["B08"]) / (df["B11"] + df["B08"] + eps)
    return df

def build_sample_map():
    points = load_sample_points()
    m = folium.Map(location=[-7.7, 112.5], zoom_start=7, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite", name="Google Satellite", overlay=False, control=True,
    ).add_to(m)
    for class_key, samples in points.items():
        fg = folium.FeatureGroup(name=DISPLAY_NAMES.get(class_key, class_key), show=True)
        for point in samples:
            lat, lon = point.get("lat"), point.get("lon")
            if lat is None or lon is None:
                continue
            popup = (
                f"<b>{point.get('title', DISPLAY_NAMES.get(class_key, class_key))}</b><br>"
                f"Koordinat: {lat:.5f}, {lon:.5f}<br>"
                f"NDVI: {point.get('ndvi', float('nan')):.4f}<br>"
                f"NDWI: {point.get('ndwi', float('nan')):.4f}<br>"
                f"MNDWI: {point.get('mndwi', float('nan')):.4f}<br>"
                f"NDBI: {point.get('ndbi', float('nan')):.4f}"
            )
            folium.CircleMarker(
                location=[lat, lon], radius=5, color="white", weight=1,
                fill=True, fill_opacity=0.9,
                fill_color=COLORS.get(class_key, "#777777"),
                tooltip=DISPLAY_NAMES.get(class_key, class_key),
                popup=folium.Popup(popup, max_width=280),
            ).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m

st.title("🛰️ Klasifikasi Tutupan Lahan (LULC) Jawa Timur")
st.caption("Sentinel-2A • Indeks spektral • Machine Learning • WebGIS interaktif")
st.info(
    "Model klasifikasi dimuat otomatis dari file best_model_lulc.joblib. "
    "Unggah CSV baru untuk melakukan prediksi. Peta titik contoh bersifat representatif, "
    "bukan raster klasifikasi penuh."
)

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
        st.caption("Untuk menampilkan peta HTML hasil notebook, letakkan peta_lulc_jatim_webgis.html di folder aplikasi.")
    if raster_path.exists():
        st.subheader("Raster hasil klasifikasi")
        st.image(Image.open(raster_path), use_container_width=True)
    else:
        st.warning("File peta_raster_lulc_jatim.png belum tersedia. Letakkan file raster hasil notebook di folder aplikasi jika ingin menampilkannya.")

with tab_model:
    st.subheader("Prediksi tutupan lahan dari CSV")
    st.write("Model tersimpan dimuat otomatis. CSV prediksi harus memiliki lima kolom band: B02, B03, B04, B08, B11.")
    if not MODEL_PATH.exists():
        st.error(f"Model tidak ditemukan: {MODEL_PATH.name}. Letakkan best_model_lulc.joblib di folder yang sama dengan app.py.")
        st.stop()
    try:
        model = load_saved_model()
        st.success("Model best_model_lulc.joblib berhasil dimuat.")
    except Exception as exc:
        st.error(f"Gagal memuat model tersimpan: {exc}")
        st.stop()

    pred_file = st.file_uploader("Unggah CSV untuk diprediksi", type=["csv"], key="predict_csv")
    if pred_file is not None:
        try:
            source_df = pd.read_csv(pred_file)
            prepared = add_indices(source_df)
            valid = prepared.dropna(subset=FEATURES).copy()
            if valid.empty:
                st.warning("Tidak ada baris valid. Periksa apakah nilai pada kolom band berupa angka.")
            else:
                predictions = model.predict(valid[FEATURES])
                valid["prediksi_label"] = predictions
                valid["prediksi_kelas"] = [CLASS_NAMES.get(int(x), str(x)) for x in predictions]
                st.success(f"Prediksi selesai untuk {len(valid)} baris data.")
                st.dataframe(valid.head(100), use_container_width=True)
                st.download_button(
                    "Unduh hasil prediksi CSV",
                    data=valid.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_lulc.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Gagal melakukan prediksi: {exc}")

with tab_about:
    st.subheader("Ringkasan metode")
    st.markdown("""
- **Wilayah studi:** Provinsi Jawa Timur.
- **Sumber citra:** Sentinel-2 L2A melalui OpenEO.
- **Band:** B02, B03, B04, B08, B11.
- **Indeks spektral:** NDVI, NDWI, MNDWI, NDBI.
- **Kelas:** sawah, bangunan, danau, lahan hijau, mangrove, laut.
- **Model:** dimuat dari file `best_model_lulc.joblib` yang sudah dilatih.

Aplikasi ini tidak mengambil data OpenEO secara otomatis ketika dibuka. Peta HTML dan raster hasil notebook perlu disediakan sebagai file terpisah jika ingin ditampilkan.
""")
