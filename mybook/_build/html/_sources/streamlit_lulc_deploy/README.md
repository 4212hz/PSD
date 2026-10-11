# Deploy WebGIS LULC Jawa Timur ke Streamlit

Aplikasi ini diadaptasi dari `PSD_klasifikasi_dan_webgis.ipynb`: fitur Sentinel-2, indeks NDVI/NDWI/MNDWI/NDBI, tiga pilihan model (jika dependensinya tersedia), prediksi CSV, dan peta Folium dengan titik sampel representatif.

## Isi folder
- `app.py`: aplikasi Streamlit.
- `sample_points.json`: titik contoh yang diambil dari notebook.
- `requirements.txt`: dependensi Python.
- Opsional: `peta_lulc_jatim_webgis.html` dan `peta_raster_lulc_jatim.png` untuk menampilkan hasil peta dari notebook.

## A. Jalankan di komputer sendiri
1. Instal Python 3.10 atau 3.11 dari https://www.python.org/downloads/ (centang **Add Python to PATH** di Windows).
2. Ekstrak ZIP ini.
3. Buka Terminal/Command Prompt pada folder yang berisi `app.py`.
4. Buat virtual environment:
   - Windows: `py -m venv .venv` lalu `.venv\\Scripts\\activate`
   - macOS/Linux: `python3 -m venv .venv` lalu `source .venv/bin/activate`
5. Instal paket: `python -m pip install --upgrade pip` kemudian `pip install -r requirements.txt`.
6. Jalankan aplikasi: `streamlit run app.py`.
7. Browser akan membuka alamat lokal (biasanya `http://localhost:8501`).

## B. Siapkan data dari notebook
- Dataset training: `dataset_sentinel2_jatim.csv` atau `dataset_sentinel2_jatim_features.csv`. Kolom minimal untuk training adalah `B02`, `B03`, `B04`, `B08`, `B11`, dan `label` atau `label_teks`.
- Untuk peta lengkap, salin `peta_lulc_jatim_webgis.html` ke folder yang sama dengan `app.py`. Aplikasi akan menampilkannya otomatis.
- Untuk raster, salin `peta_raster_lulc_jatim.png` ke folder yang sama dengan `app.py`.
- Notebook menyebut file ZIP kelas (sawah.zip, bangunan.zip, danau.zip, lahan_hijau.zip, mangrove.zip, laut.zip) untuk tahap pengumpulan data. File tersebut tidak dibutuhkan saat aplikasi hanya menampilkan peta yang sudah jadi atau memproses CSV.
- Jangan unggah kredensial OpenEO ke repositori publik. Proses ekstraksi OpenEO dilakukan di notebook, bukan saat aplikasi Streamlit dijalankan.

## C. Deploy gratis ke Streamlit Community Cloud
1. Buat akun/login GitHub di https://github.com/.
2. Buat repository baru, misalnya `webgis-lulc-jatim`. Pilih **Public** jika ingin paling mudah untuk deployment gratis, atau gunakan opsi repository yang didukung akun Anda.
3. Unggah file `app.py`, `sample_points.json`, `requirements.txt`, serta file HTML/raster/data yang memang ingin dipakai. Hindari mengunggah data sensitif atau rahasia.
4. Buka https://share.streamlit.io/ dan masuk dengan GitHub.
5. Klik **Create app** / **Deploy an app**.
6. Pilih repository, branch (umumnya `main`), dan file utama `app.py`.
7. Klik **Deploy**. Tunggu instalasi paket dan build selesai; proses pertama dapat memerlukan beberapa menit.
8. Setelah aplikasi terbuka, uji peta, unggah CSV training, latih model, lalu coba prediksi dengan CSV yang memiliki lima kolom band.
9. Untuk pembaruan, commit/push perubahan ke GitHub; Streamlit Cloud biasanya membangun ulang aplikasi otomatis.

## D. Jika deployment gagal
- Buka halaman aplikasi di Streamlit Cloud, lalu lihat **Manage app → Logs**.
- Jika paket terlalu berat atau build time-out, hapus `xgboost`/`lightgbm` dari `requirements.txt`; aplikasi akan tetap memiliki Random Forest dan otomatis menyembunyikan opsi algoritma yang tidak terpasang.
- Jika peta tidak muncul, periksa bahwa `peta_lulc_jatim_webgis.html` benar-benar ada di root folder dan HTML menggunakan tile map yang dapat diakses dari internet.
- Jika prediksi memberi peringatan, latih model dahulu pada tab **Klasifikasi Model** dalam sesi aplikasi yang sama.
- Jika dataset terlalu kecil untuk split stratified, pastikan setiap kelas memiliki setidaknya dua sampel.

## Catatan
Akurasi 93,67% yang tertulis pada bagian visualisasi notebook tidak dihitung ulang atau dijamin oleh aplikasi ini. Nilai aktual akan ditampilkan setelah model dilatih ulang pada dataset yang Anda unggah.
