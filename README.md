# AI Stock Predictor BEI — Free Edition

Dashboard edukasi untuk menganalisis saham BEI dengan data historis Yahoo Finance (`KODE.JK`).

## Fitur awal
- Grafik candlestick, SMA20, SMA50
- RSI, MACD, ATR, volume
- Sinyal berbasis aturan: BUY / HOLD-WAIT / SELL-AVOID
- Area entry indikatif, stop loss, target 1 dan target 2
- Snapshot indeks global, USD/IDR, emas, dan minyak
- Cache data 15 menit

## Catatan penting
Versi pertama belum memakai machine learning dan bukan penasihat investasi. Data dari Yahoo Finance melalui yfinance bukan feed resmi BEI dan dapat tertunda/tidak lengkap. Data global saat ini ditampilkan sebagai konteks, belum memengaruhi skor sinyal. Selalu verifikasi harga dan informasi emiten di sumber resmi/aplikasi sekuritas.

## Jalankan di komputer
1. Install Python 3.10 atau lebih baru.
2. Buka terminal di folder proyek.
3. Buat virtual environment (opsional tetapi disarankan):
   `python -m venv .venv`
4. Aktifkan environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
5. Install paket: `pip install -r requirements.txt`
6. Jalankan: `streamlit run app.py`

## Deploy gratis dengan GitHub + Streamlit Community Cloud
1. Buat repository GitHub baru dan upload `app.py`, `requirements.txt`, dan `README.md`.
2. Buka https://share.streamlit.io/ dan masuk dengan GitHub.
3. Pilih **Create app**, repository, branch, dan file utama `app.py`.
4. Deploy. Jika layanan gratis sedang membatasi kapasitas atau kebijakannya berubah, ikuti ketentuan terbaru yang ditampilkan Streamlit.

## Tahap pengembangan berikutnya
1. Tambahkan sumber berita/RSS dan analisis sentimen.
2. Masukkan faktor global ke dalam skor dengan uji korelasi dan lag yang jelas.
3. Buat backtesting dengan biaya transaksi, slippage, dan aturan fraksi harga BEI.
4. Evaluasi model machine learning secara walk-forward tanpa data leakage.
5. Tambahkan watchlist dan notifikasi.
