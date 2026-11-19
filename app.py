import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="AI Stock Predictor BEI", page_icon="📈", layout="wide")

GLOBAL_ASSETS = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "USD/IDR": "IDR=X",
    "Emas": "GC=F",
    "Minyak WTI": "CL=F",
}

@st.cache_data(ttl=900, show_spinner=False)
def load_history(ticker: str, period: str = "1y"):
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    return df.dropna(subset=["Close"])

def add_indicators(df):
    d = df.copy()
    close = d["Close"]
    d["SMA20"] = close.rolling(20).mean()
    d["SMA50"] = close.rolling(50).mean()
    d["EMA12"] = close.ewm(span=12, adjust=False).mean()
    d["EMA26"] = close.ewm(span=26, adjust=False).mean()
    d["MACD"] = d["EMA12"] - d["EMA26"]
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))
    prev = d["Close"].shift(1)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev).abs(),
        (d["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()
    d["BB_Mid"] = close.rolling(20).mean()
    std = close.rolling(20).std()
    d["BB_Upper"] = d["BB_Mid"] + 2 * std
    d["BB_Lower"] = d["BB_Mid"] - 2 * std
    d["Vol_Avg20"] = d["Volume"].rolling(20).mean()
    return d

def analyze(d):
    last = d.iloc[-1]
    score, reasons = 0, []
    if pd.notna(last["SMA20"]) and last["Close"] > last["SMA20"]:
        score += 1; reasons.append("Harga di atas SMA20 (momentum jangka pendek positif).")
    else:
        score -= 1; reasons.append("Harga di bawah SMA20 (momentum jangka pendek lemah).")
    if pd.notna(last["SMA50"]) and last["Close"] > last["SMA50"]:
        score += 1; reasons.append("Harga di atas SMA50 (tren menengah positif).")
    else:
        score -= 1; reasons.append("Harga di bawah SMA50 (tren menengah lemah).")
    if pd.notna(last["MACD"]) and pd.notna(last["MACD_Signal"]):
        if last["MACD"] > last["MACD_Signal"]:
            score += 1; reasons.append("MACD berada di atas garis sinyal.")
        else:
            score -= 1; reasons.append("MACD berada di bawah garis sinyal.")
    if pd.notna(last["RSI"]):
        if 50 <= last["RSI"] <= 68:
            score += 1; reasons.append("RSI menunjukkan momentum positif tanpa kondisi terlalu jenuh beli.")
        elif last["RSI"] > 75:
            score -= 1; reasons.append("RSI tinggi: risiko kondisi jenuh beli.")
        elif last["RSI"] < 30:
            reasons.append("RSI rendah: potensi pantulan ada, tetapi tren turun tetap perlu diwaspadai.")
        else:
            reasons.append("RSI belum menunjukkan momentum beli yang kuat.")
    if pd.notna(last["Vol_Avg20"]) and last["Vol_Avg20"] > 0:
        if last["Volume"] > last["Vol_Avg20"] * 1.2 and last["Close"] > last["Open"]:
            score += 1; reasons.append("Volume di atas rata-rata disertai candle naik.")
    if score >= 3:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL / AVOID"
    else:
        signal = "HOLD / WAIT"
    price = float(last["Close"])
    atr = float(last["ATR"]) if pd.notna(last["ATR"]) else price * 0.03
    return {
        "signal": signal, "score": score, "price": price,
        "entry_low": price - 0.25 * atr, "entry_high": price + 0.10 * atr,
        "stop": price - 1.5 * atr, "target1": price + 2 * atr,
        "target2": price + 3 * atr, "rsi": float(last["RSI"]) if pd.notna(last["RSI"]) else np.nan,
        "reasons": reasons
    }

st.title("📈 AI Stock Predictor BEI")
st.caption("Edisi gratis • Analisis teknikal berbasis aturan • Bukan rekomendasi investasi")
with st.sidebar:
    st.header("Pengaturan")
    symbol = st.text_input("Kode saham BEI", value="BBCA").strip().upper()
    period_label = st.selectbox("Riwayat data", ["6 bulan", "1 tahun", "2 tahun", "5 tahun"], index=1)
    period = {"6 bulan":"6mo", "1 tahun":"1y", "2 tahun":"2y", "5 tahun":"5y"}[period_label]
    st.caption("Masukkan kode saja, misalnya BBCA. Aplikasi menambahkan .JK.")
    refresh = st.button("Muat ulang data")

if refresh:
    st.cache_data.clear()

ticker = symbol if "." in symbol else f"{symbol}.JK"
with st.spinner(f"Mengambil data {ticker}..."):
    raw = load_history(ticker, period)

if raw.empty or len(raw) < 60:
    st.error("Data saham tidak ditemukan atau belum cukup untuk analisis. Periksa kode saham dan coba lagi nanti.")
    st.info("Sumber data menggunakan Yahoo Finance melalui yfinance. Ketersediaan dan ketepatan waktu data tidak dijamin.")
    st.stop()

data = add_indicators(raw)
result = analyze(data)
last_date = data.index[-1]
age_days = (pd.Timestamp.now(tz=last_date.tz) - last_date).days if last_date.tz else (pd.Timestamp.now().tz_localize(None) - last_date).days

c1,c2,c3,c4 = st.columns(4)
c1.metric("Harga terakhir", f"Rp {result['price']:,.0f}")
c2.metric("Sinyal sistem", result["signal"])
c3.metric("RSI 14", f"{result['rsi']:.1f}" if pd.notna(result["rsi"]) else "—")
c4.metric("Skor indikator", f"{result['score']}/5")

st.caption(f"Ticker: {ticker} • Tanggal candle terakhir: {last_date.strftime('%d-%m-%Y')} • Perkiraan umur data: {age_days} hari • Cache data: 15 menit")
if age_days > 4:
    st.warning("Data terakhir tampak lebih lama dari 4 hari kalender. Jangan gunakan sinyal ini untuk keputusan transaksi sebelum memverifikasi harga terbaru di aplikasi sekuritas.")

st.subheader("Rencana transaksi indikatif")
p1,p2,p3,p4 = st.columns(4)
p1.metric("Area entry", f"Rp {result['entry_low']:,.0f} – {result['entry_high']:,.0f}")
p2.metric("Stop loss", f"Rp {result['stop']:,.0f}")
p3.metric("Target 1", f"Rp {result['target1']:,.0f}")
p4.metric("Target 2", f"Rp {result['target2']:,.0f}")
st.caption("Level dihitung dari ATR historis, bukan jaminan akan tersentuh. Sesuaikan dengan fraksi harga, likuiditas, biaya transaksi, dan toleransi risiko.")

st.subheader("Grafik harga dan moving average")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name="Harga"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], name="SMA20", line=dict(width=1.5)))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], name="SMA50", line=dict(width=1.5)))
fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(fig, use_container_width=True)

left,right = st.columns(2)
with left:
    st.subheader("Alasan sinyal")
    for reason in result["reasons"]:
        st.write("• " + reason)
with right:
    st.subheader("Momentum")
    mom = go.Figure()
    mom.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI"))
    mom.add_hline(y=70, line_dash="dash")
    mom.add_hline(y=30, line_dash="dash")
    mom.update_layout(height=280, yaxis_range=[0,100], margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(mom, use_container_width=True)

st.subheader("Kondisi pasar global")
cols = st.columns(4)
for i, (name, code) in enumerate(GLOBAL_ASSETS.items()):
    try:
        g = load_history(code, "3mo")
        if not g.empty and len(g) >= 2:
            latest = float(g["Close"].iloc[-1])
            prev = float(g["Close"].iloc[-2])
            change = (latest / prev - 1) * 100 if prev else 0
            cols[i % 4].metric(name, f"{latest:,.2f}", f"{change:+.2f}%")
        else:
            cols[i % 4].metric(name, "Data tidak tersedia")
    except Exception:
        cols[i % 4].metric(name, "Gagal dimuat")

st.subheader("Data historis terbaru")
st.dataframe(data[["Open","High","Low","Close","Volume","RSI","MACD","SMA20","SMA50"]].tail(15), use_container_width=True)

with st.expander("Penting: batasan dan penggunaan yang aman"):
    st.markdown("""
    - Versi awal ini **belum menggunakan model machine learning**. Istilah AI Stock Predictor adalah nama produk; sinyal saat ini berasal dari aturan indikator teknikal yang transparan.
    - Data Yahoo Finance/yfinance dapat tertunda, tidak lengkap, berubah, atau gagal diakses. Bukan feed resmi BEI dan bukan data tick real-time.
    - Indeks global ditampilkan sebagai konteks, belum digabungkan ke skor sinyal saham. Integrasi sentimen berita dan model prediksi akan menjadi tahap pengembangan berikutnya.
    - Uji strategi dengan data historis (backtesting) dan akun simulasi sebelum mempertimbangkan transaksi nyata.
    - Tidak ada sinyal yang menjamin keuntungan. Keputusan akhir dan risiko tetap pada pengguna.
    """)
st.caption("Dibuat untuk edukasi dan riset. Periksa ulang harga, aksi korporasi, likuiditas, dan informasi emiten melalui sumber resmi sebelum bertransaksi.")
