
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="AI Stock Predictor BEI V2",
    page_icon="📈",
    layout="wide"
)

WATCHLIST = [
    "BBCA", "BBRI", "BMRI", "BBNI", "BRIS",
    "TLKM", "ASII", "ANTM", "INCO", "MDKA",
    "ADRO", "PTBA", "ITMG", "UNTR", "ICBP",
    "INDF", "AMRT", "CPIN", "GOTO", "ACES",
    "UNVR", "KLBF", "PGAS", "MEDC", "SMGR"
]

st.title("📈 AI STOCK PREDICTOR BEI")
st.caption("VERSION 2 | Auto Scanner + Manual Analysis")

st.info(
    "Analisis berbasis indikator teknikal dan konteks pasar global. "
    "Bukan jaminan keuntungan atau rekomendasi investasi."
)

@st.cache_data(ttl=300)
def get_data(ticker, period="6mo"):
    return yf.download(
        ticker + ".JK",
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

def analyze(ticker):
    df = get_data(ticker)

    if df is None or df.empty or len(df) < 55:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["ATR"] = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1).rolling(14).mean()

    last = df.iloc[-1]
    price = float(close.iloc[-1])
    score = 0

    if price > last["SMA20"]:
        score += 1
    if last["SMA20"] > last["SMA50"]:
        score += 1
    if last["MACD"] > last["Signal"]:
        score += 1
    if 50 <= last["RSI"] <= 70:
        score += 1
    if volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
        score += 1

    if score >= 4:
        signal = "BUY"
    elif score <= 1:
        signal = "SELL"
    else:
        signal = "HOLD"

    atr = float(last["ATR"])
    return {
        "ticker": ticker,
        "price": price,
        "signal": signal,
        "score": score,
        "rsi": float(last["RSI"]),
        "sma20": float(last["SMA20"]),
        "sma50": float(last["SMA50"]),
        "atr": atr,
        "entry": price,
        "stop": price - (1.5 * atr),
        "target": price + (3 * atr),
        "data": df
    }

def rupiah(value):
    return "Rp {:,.0f}".format(value)

mode = st.sidebar.radio(
    "PILIH MODE ANALISIS",
    ["AUTO SCANNER", "MANUAL ANALYSIS"]
)

st.sidebar.caption("Data: Yahoo Finance | Dapat tertunda")

# KONTEKS PASAR GLOBAL
st.subheader("🌍 KONTEKS PASAR GLOBAL")

global_symbols = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "NIKKEI": "^N225",
    "HANG SENG": "^HSI",
    "USD/IDR": "IDR=X",
    "GOLD": "GC=F",
    "WTI OIL": "CL=F"
}

global_data = []
for name, symbol in global_symbols.items():
    try:
        data = yf.download(
            symbol, period="5d", interval="1d",
            progress=False, auto_adjust=True
        )
        if not data.empty:
            price = float(data["Close"].iloc[-1].squeeze())
            global_data.append((name, price))
    except Exception:
        pass

if global_data:
    cols = st.columns(4)
    for i, (name, price) in enumerate(global_data):
        cols[i % 4].metric(name, f"{price:,.2f}")
else:
    st.warning("Data pasar global belum tersedia.")

st.divider()

# AUTO SCANNER
if mode == "AUTO SCANNER":
    st.subheader("🔎 AUTO SCANNER SAHAM BEI")
    st.write("Memindai 25 saham pilihan secara otomatis.")

    if st.button("🚀 MULAI SCAN SAHAM", type="primary"):
        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, ticker in enumerate(WATCHLIST):
            status.write(f"Memeriksa {ticker}...")
            try:
                result = analyze(ticker)
                if result:
                    results.append({
                        "Kode": ticker,
                        "Harga": result["price"],
                        "Sinyal": result["signal"],
                        "Skor": result["score"],
                        "RSI": round(result["rsi"], 2),
                        "Potensi Target": result["target"]
                    })
            except Exception:
                pass

            progress.progress((i + 1) / len(WATCHLIST))

        status.success("Pemindaian selesai!")

        if results:
            result_df = pd.DataFrame(results)
            result_df = result_df.sort_values(
                "Skor", ascending=False
            )
            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                "📥 Download Hasil Scanner CSV",
                result_df.to_csv(index=False),
                file_name="hasil_scanner_saham.csv",
                mime="text/csv"
            )

            st.subheader("Saham dengan Skor Tertinggi")
            st.dataframe(
                result_df.head(5),
                use_container_width=True
            )
        else:
            st.error("Data saham belum berhasil diperoleh.")

# MANUAL ANALYSIS
else:
    st.subheader("📊 ANALISIS SAHAM MANUAL")

    ticker = st.text_input(
        "Masukkan kode saham BEI",
        value="BBCA"
    ).upper().strip()

    if st.button("ANALISIS SAHAM", type="primary"):
        with st.spinner("Menganalisis saham..."):
            try:
                result = analyze(ticker)

                if result is None:
                    st.error("Data tidak tersedia atau belum cukup.")
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Harga Terakhir", rupiah(result["price"]))
                    c2.metric("Sinyal", result["signal"])
                    c3.metric("Skor Teknikal", f'{result["score"]}/5')

                    st.subheader("Grafik Harga dan Moving Average")

                    df = result["data"]
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name="Harga"
                    ))
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df["SMA20"],
                        name="SMA 20"
                    ))
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df["SMA50"],
                        name="SMA 50"
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Indikator Teknikal")
                    a, b, c = st.columns(3)
                    a.metric("RSI", f'{result["rsi"]:.2f}')
                    b.metric("SMA 20", rupiah(result["sma20"]))
                    c.metric("SMA 50", rupiah(result["sma50"]))

                    st.subheader("Rencana Harga Indikatif")
                    a, b, c = st.columns(3)
                    a.metric("Area Entri", rupiah(result["entry"]))
                    b.metric("Stop Loss", rupiah(result["stop"]))
                    c.metric("Target", rupiah(result["target"]))

                    st.caption(
                        "Level dihitung menggunakan ATR historis, "
                        "bukan prediksi harga yang pasti."
                    )

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

st.divider()
st.caption(
    "AI Stock Predictor BEI V2 | "
    "Analisis teknikal berbasis aturan, bukan machine learning. "
    "Data Yahoo Finance dapat tertunda atau tidak lengkap. "
    "Bukan nasihat investasi."
)
st.caption("Diperbarui: " + datetime.now().strftime("%d-%m-%Y %H:%M"))
