import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="AI Stock Predictor BEI v2", page_icon="📈", layout="wide")
WATCHLIST = ["BBCA.JK","BBRI.JK","BMRI.JK","BBNI.JK","BRIS.JK","TLKM.JK","ASII.JK","ANTM.JK","INCO.JK","MDKA.JK","ADRO.JK","PTBA.JK","ITMG.JK","UNTR.JK","ICBP.JK","INDF.JK","AMRT.JK","CPIN.JK","GOTO.JK","ACES.JK","UNVR.JK","KLBF.JK","PGAS.JK","MEDC.JK","SMGR.JK"]
GLOBAL = {"S&P 500":"^GSPC","Nasdaq":"^IXIC","Nikkei 225":"^N225","Hang Seng":"^HSI","USD/IDR":"IDR=X","Emas":"GC=F","Minyak WTI":"CL=F"}

@st.cache_data(ttl=900)
def get_data(ticker, period="1y"):
    try:
        x=yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        return x.rename(columns=str.title).dropna(subset=["Close"]) if x is not None and not x.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def calc(x):
    d=x.copy(); c=d.Close
    d["SMA20"]=c.rolling(20).mean(); d["SMA50"]=c.rolling(50).mean()
    d["MACD"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    d["MACDsig"]=d.MACD.ewm(span=9,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    d["RSI"]=100-100/(1+gain/loss.replace(0,np.nan))
    prev=c.shift(1)
    tr=pd.concat([d.High-d.Low,(d.High-prev).abs(),(d.Low-prev).abs()],axis=1).max(axis=1)
    d["ATR"]=tr.rolling(14).mean(); d["VMA20"]=d.Volume.rolling(20).mean()
    return d

@st.cache_data(ttl=900)
def global_data():
    rows=[]
    for name,t in GLOBAL.items():
        x=get_data(t,"3mo")
        if len(x)>=2:
            a=float(x.Close.iloc[-1]); b=float(x.Close.iloc[-2])
            rows.append([name,a,(a/b-1)*100 if b else np.nan])
        else: rows.append([name,np.nan,np.nan])
    return pd.DataFrame(rows,columns=["Indikator","Terakhir","Perubahan harian %"])

def score_global(g):
    s=0
    for name in ["S&P 500","Nasdaq","Nikkei 225","Hang Seng"]:
        v=g.loc[g.Indikator==name,"Perubahan harian %"]
        if len(v) and pd.notna(v.iloc[0]): s += 1 if v.iloc[0]>0 else -1 if v.iloc[0]<0 else 0
    v=g.loc[g.Indikator=="USD/IDR","Perubahan harian %"]
    if len(v) and pd.notna(v.iloc[0]): s += -1 if v.iloc[0]>0 else 1 if v.iloc[0]<0 else 0
    return s

def analyze(ticker, gbias):
    x=get_data(ticker,"1y")
    if len(x)<60: return None
    d=calc(x); z=d.iloc[-1]; s=0; why=[]
    for ok,yes,no in [
        (pd.notna(z.SMA20) and z.Close>z.SMA20,"Harga di atas SMA20","Harga di bawah SMA20"),
        (pd.notna(z.SMA50) and z.Close>z.SMA50,"Harga di atas SMA50","Harga di bawah SMA50"),
        (pd.notna(z.MACD) and pd.notna(z.MACDsig) and z.MACD>z.MACDsig,"MACD positif","MACD negatif")]:
        s += 1 if ok else -1; why.append(yes if ok else no)
    if pd.notna(z.RSI):
        if 50<=z.RSI<=68: s+=1; why.append("RSI mendukung momentum")
        elif z.RSI>75: s-=1; why.append("RSI tinggi, risiko jenuh beli")
        elif z.RSI<30: why.append("RSI rendah, waspadai tren turun")
    if pd.notna(z.VMA20) and z.VMA20>0 and z.Volume>1.2*z.VMA20 and z.Close>z.Open: s+=1; why.append("Volume meningkat")
    s += 1 if gbias>=3 else -1 if gbias<=-3 else 0
    price=float(z.Close); atr=float(z.ATR) if pd.notna(z.ATR) else price*.03
    sig="BUY WATCH" if s>=3 else "SELL / AVOID" if s<=-2 else "HOLD / WAIT"
    return {"Kode":ticker.replace(".JK",""),"Harga":price,"Skor":s,"Sinyal":sig,"RSI":float(z.RSI) if pd.notna(z.RSI) else np.nan,
            "Entry bawah":price-.25*atr,"Entry atas":price+.1*atr,"Stop loss":price-1.5*atr,"Target":price+2*atr,
            "Tanggal data":d.index[-1],"data":d,"alasan":why}

st.title("📈 AI Stock Predictor BEI")
st.caption("Versi 2 • Auto Scanner + Manual Analysis • Gratis • Untuk edukasi dan riset")
mode=st.sidebar.radio("Mode",["AUTO SCANNER","MANUAL ANALYSIS"])
st.sidebar.caption("Sumber data pihak ketiga dapat tertunda/tidak lengkap.")
g=global_data(); bias=score_global(g)
st.subheader("Kondisi pasar global")
st.dataframe(g.style.format({"Terakhir":"{:,.2f}","Perubahan harian %":"{:+.2f}%"}),use_container_width=True)
st.caption(f"Skor konteks global sederhana: {bias:+d}. Ini bukan bukti hubungan sebab-akibat.")
if mode=="AUTO SCANNER":
    st.subheader("Auto Scanner saham BEI")
    st.warning("Daftar awal hanya 25 saham pemantauan, bukan seluruh emiten BEI. Cakupan dapat diperluas setelah tersedia daftar emiten dan data yang memadai.")
    selected=st.multiselect("Pilih saham untuk dipindai",WATCHLIST,default=WATCHLIST,format_func=lambda x:x.replace(".JK",""))
    if st.button("Jalankan pemindaian",type="primary"):
        results=[]; bar=st.progress(0)
        with ThreadPoolExecutor(max_workers=5) as pool:
            fs=[pool.submit(analyze,t,bias) for t in selected]
            for i,f in enumerate(as_completed(fs),1):
                try:
                    r=f.result()
                    if r: results.append({k:v for k,v in r.items() if k not in ["data","alasan","Tanggal data"]})
                except Exception: pass
                bar.progress(i/max(len(fs),1))
        if results: st.session_state["scan"]=pd.DataFrame(results).sort_values("Skor",ascending=False)
        else: st.warning("Data tidak tersedia atau belum cukup. Coba lagi nanti.")
    if "scan" in st.session_state:
        out=st.session_state["scan"]
        st.dataframe(out,use_container_width=True)
        st.download_button("Unduh hasil CSV",out.to_csv(index=False).encode(), "scanner_bei.csv","text/csv")
else:
    st.subheader("Analisis manual")
    sym=st.text_input("Kode saham BEI", "BBCA").strip().upper()
    t=sym if "." in sym else sym+".JK"
    r=analyze(t,bias)
    if not r: st.error("Data tidak ditemukan atau belum cukup. Coba BBCA, BBRI, BMRI, TLKM, atau ANTM.")
    else:
        a,b,c,dcol=st.columns(4)
        a.metric("Harga terakhir",f"Rp {r['Harga']:,.0f}"); b.metric("Sinyal",r["Sinyal"])
        c.metric("Skor",r["Skor"]); dcol.metric("RSI",f"{r['RSI']:.1f}" if pd.notna(r["RSI"]) else "—")
        e,f,h,i=st.columns(4)
        e.metric("Area entry",f"{r['Entry bawah']:,.0f}–{r['Entry atas']:,.0f}")
        f.metric("Stop loss",f"Rp {r['Stop loss']:,.0f}")
        h.metric("Target",f"Rp {r['Target']:,.0f}")
        i.metric("Tanggal data",r["Tanggal data"].strftime("%d-%m-%Y"))
        d=r["data"]; fig=go.Figure()
        fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name="Harga"))
        fig.add_trace(go.Scatter(x=d.index,y=d.SMA20,name="SMA20")); fig.add_trace(go.Scatter(x=d.index,y=d.SMA50,name="SMA50"))
        fig.update_layout(height=480,xaxis_rangeslider_visible=False); st.plotly_chart(fig,use_container_width=True)
        st.subheader("Alasan sinyal")
        for q in r["alasan"]: st.write("• "+q)
with st.expander("Batasan penting"):
    st.markdown("""
    - Ini skor berbasis aturan teknikal, **belum machine learning** dan tidak menjamin prediksi.
    - Auto Scanner baru memindai 25 saham pilihan, bukan seluruh BEI.
    - Yahoo Finance/yfinance bukan feed resmi BEI; data dapat tertunda, tidak lengkap, atau gagal.
    - Skor global hanya penyesuaian sederhana dari perubahan harian indeks dan USD/IDR; belum memasukkan berita atau kalender ekonomi.
    - Level entry/stop/target berdasarkan ATR historis dan bersifat indikatif. Verifikasi harga, likuiditas, fraksi harga, biaya, serta aksi korporasi.
    - Lakukan backtesting dan paper trading sebelum transaksi nyata.
    """)
st.caption("Bukan nasihat investasi. Keputusan dan risiko transaksi tetap pada pengguna.")
