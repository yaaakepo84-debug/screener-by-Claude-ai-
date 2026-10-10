import math

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Screener Day Trade IDX", page_icon="📈", layout="wide")

DEFAULT_WATCHLIST = (
    "BBCA BBRI BMRI BBNI TLKM ASII GOTO BRIS ANTM MDKA ADRO BUKA PGEO ACES "
    "INCO SMGR UNTR ICBP INDF KLBF AMRT CPIN MAPI PTBA ITMG MEDC AKRA EXCL ISAT TOWR"
)

# d = default filter, w = bobot skor
PRESETS = {
    "Breakout lanjutan": dict(
        d=dict(price=100, val=10.0, rvol=1.5, chg=1.0, chg_max=100.0, pos=0.7, near=3.0, atr=1.5, trend=True),
        w=dict(rvol=25, chg=15, pos=20, near=25, liq=10, atr=5)),
    "Volume + close kuat": dict(
        d=dict(price=100, val=10.0, rvol=2.0, chg=2.0, chg_max=100.0, pos=0.8, near=8.0, atr=1.5, trend=True),
        w=dict(rvol=35, chg=20, pos=25, near=5, liq=10, atr=5)),
    "Pullback sehat": dict(
        d=dict(price=100, val=10.0, rvol=0.5, chg=-3.0, chg_max=0.5, pos=0.0, near=8.0, atr=1.5, trend=True),
        w=dict(rvol=10, chg=0, pos=10, near=30, liq=25, atr=25)),
}


for _v in PRESETS.values():
    _v["d"].setdefault("pmax", 1_000_000.0)
    _v["d"].setdefault("rs", -100.0)
PRESETS = {
    "Saham di bawah 100": dict(
        d=dict(price=50, pmax=100.0, val=3.0, rvol=2.0, chg=2.0, chg_max=100.0, pos=0.75, near=5.0,
               atr=3.0, trend=False, rs=1.0),
        w=dict(rvol=30, chg=20, pos=25, near=10, liq=10, atr=5)),
    **PRESETS,
}


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def tick(p):  # fraksi harga (perkiraan, cek ke broker)
    return 1 if p < 200 else 2 if p < 500 else 5 if p < 2000 else 10 if p < 5000 else 25


def up(p):
    t = tick(p)
    return math.ceil(p / t) * t


def down(p):
    t = tick(p)
    return math.floor(p / t) * t


@st.cache_data(ttl=600, show_spinner=False)
def load_data(codes: tuple):
    tickers = [c + ".JK" for c in codes]
    raw = yf.download(tickers, period="6mo", interval="1d", group_by="ticker",
                      auto_adjust=False, progress=False, threads=True)
    rows, missing = [], []
    for c in codes:
        try:
            d = raw[c + ".JK"] if isinstance(raw.columns, pd.MultiIndex) else raw
            d = d.dropna(subset=["Close"])
            if len(d) < 55:
                missing.append(c)
                continue
            last, pc = d.iloc[-1], d["Close"].iloc[-2]
            avg_vol = d["Volume"].iloc[-21:-1].mean()
            if pc <= 0 or avg_vol <= 0:
                missing.append(c)
                continue
            prev_close = d["Close"].shift(1)
            tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev_close).abs(),
                            (d["Low"] - prev_close).abs()], axis=1).max(axis=1)
            atr = tr.iloc[-14:].mean()
            high20 = d["High"].iloc[-21:-1].max()
            rng = last["High"] - last["Low"]
            rows.append(dict(
                Kode=c, Tanggal=d.index[-1].date(), Close=float(last["Close"]),
                High=float(last["High"]), Low=float(last["Low"]), ATR=float(atr),
                chg=(last["Close"] - pc) / pc * 100,
                rvol=last["Volume"] / avg_vol,
                pos=(last["Close"] - last["Low"]) / rng if rng > 0 else 0.5,
                near=(high20 - last["Close"]) / high20 * 100,
                atr_pct=atr / last["Close"] * 100,
                trend=bool(last["Close"] > d["Close"].iloc[-20:].mean()),
                val=last["Close"] * last["Volume"] / 1e9,
            ))
        except Exception:
            missing.append(c)
    return pd.DataFrame(rows), missing


@st.cache_data(ttl=600, show_spinner=False)
def get_ihsg():
    try:
        c = yf.Ticker("^JKSE").history(period="10d")["Close"].dropna()
        return float((c.iloc[-1] / c.iloc[-2] - 1) * 100), c.index[-1].date()
    except Exception:
        return None, None


def add_score(df, w):
    def f(r):
        return round(
            w["rvol"] * clip(r.rvol / 3, 0, 1) + w["chg"] * clip(max(r.chg, 0) / 6, 0, 1)
            + w["pos"] * r.pos + w["near"] * clip(1 - max(r.near, 0) / 8, 0, 1)
            + w["liq"] * clip(r.val / 20, 0, 1) + w["atr"] * clip(r.atr_pct / 4, 0, 1))
    df = df.copy()
    df["Skor"] = df.apply(f, axis=1)
    return df


def make_plan(df, modal, risk_pct, alloc_pct):
    out = []
    for r in df.itertuples():
        trigger = up(r.High) + tick(r.High)               # beli jika harga tembus high hari ini
        stop = down(max(r.Low, trigger - r.ATR))          # stop di low hari ini, maksimal 1x ATR dari trigger
        if trigger - stop < 2 * tick(trigger):            # saham murah: minimal 2 fraksi harga
            stop = trigger - 2 * tick(trigger)
        risk = trigger - stop
        target = down(trigger + 2 * risk)                 # risk:reward 1:2
        skip = max(down(trigger + 0.5 * r.ATR), trigger + 2 * tick(trigger))                # jika buka di atas ini, jangan kejar
        lots = math.floor(modal * risk_pct / 100 / risk / 100)
        lots = min(lots, math.floor(modal * alloc_pct / 100 / (trigger * 100)))
        out.append(dict(Trigger=trigger, Stop=stop, Target=target, Risiko=risk / trigger * 100,
                        Batal=skip, Lot=max(lots, 0), Nilai=max(lots, 0) * trigger * 100 / 1e6))
    return pd.DataFrame(out, index=df.index)


st.title("Screener Day Trade IDX")
st.caption("Alur: malam cari kandidat (data hari bursa yang sudah selesai), pagi eksekusi sesuai rencana, "
           "jual semua di hari yang sama. Data Yahoo Finance, bukan rekomendasi beli atau jual.")

with st.sidebar:
    st.header("Strategi")
    preset = st.selectbox("Preset", list(PRESETS))
    p = PRESETS[preset]["d"]
    k = preset  # key memuat nama preset supaya filter ikut berganti
    f_price = st.number_input("Harga min (Rp)", value=float(p["price"]), key=f"pr{k}")
    f_pmax = st.number_input("Harga maks (Rp)", value=float(p["pmax"]), key=f"pm{k}")
    f_val = st.number_input("Nilai transaksi min (Rp miliar)", value=p["val"], step=1.0, key=f"va{k}")
    f_rvol = st.number_input("Volume relatif min (x)", value=p["rvol"], step=0.1, key=f"rv{k}")
    f_chg = st.number_input("Perubahan hari ini min (%)", value=p["chg"], step=0.5, key=f"ch{k}")
    f_chg_max = st.number_input("Perubahan hari ini maks (%)", value=p["chg_max"], step=0.5, key=f"cm{k}")
    f_pos = st.slider("Posisi close min (0 = low, 1 = high)", 0.0, 1.0, float(p["pos"]), 0.05, key=f"po{k}")
    f_near = st.number_input("Jarak ke high 20 hari maks (%)", value=p["near"], step=0.5, key=f"ne{k}",
                             help="Nol atau negatif berarti sudah menembus high 20 hari.")
    f_atr = st.number_input("ATR min (% dari harga)", value=p["atr"], step=0.5, key=f"at{k}",
                            help="Ukuran gerak harian rata-rata. Terlalu kecil berarti sulit untung setelah biaya.")
    f_trend = st.checkbox("Hanya yang close di atas rata-rata 20 hari", value=p["trend"], key=f"tr{k}")
    f_rs = st.number_input("Lebih kuat dari IHSG min (poin %)", value=float(p["rs"]), step=0.5, key=f"rs{k}",
                           help="Perubahan saham hari ini dikurangi perubahan IHSG. Isi -100 untuk mematikan filter.")

    st.header("Ukuran posisi")
    modal = st.number_input("Modal trading (Rp)", value=10_000_000, step=1_000_000)
    risk_pct = st.number_input("Risiko per transaksi (% modal)", value=1.0, step=0.25, min_value=0.1, max_value=5.0)
    alloc_pct = st.number_input("Maks nilai beli per saham (% modal)", value=30.0, step=5.0, min_value=5.0, max_value=100.0)

    st.header("Watchlist")
    wl = st.text_area("Kode saham (tanpa .JK)", DEFAULT_WATCHLIST, height=130)
    if st.button("Ambil data terbaru", type="primary"):
        load_data.clear()

codes = tuple(dict.fromkeys(x.strip().upper() for x in wl.replace(",", " ").split() if x.strip()))
if not codes:
    st.info("Isi watchlist di sidebar.")
    st.stop()

with st.spinner("Mengambil data..."):
    df, missing = load_data(codes)
if df.empty:
    st.error("Data tidak berhasil diambil. Coba lagi beberapa saat atau periksa kode saham.")
    st.stop()

st.write(f"Data hari bursa: **{df['Tanggal'].max()}**. Pastikan ini hari bursa terakhir; "
         "kalau masih kemarin, data hari ini belum masuk di Yahoo, tunggu lalu klik Ambil data terbaru.")

ihsg_chg, ihsg_date = get_ihsg()
df["rs"] = df["chg"] - (ihsg_chg if ihsg_chg is not None else 0.0)
if ihsg_chg is not None:
    st.write(f"IHSG ({ihsg_date}): **{ihsg_chg:+.2f}%**. Kolom 'vs IHSG' menunjukkan seberapa kuat saham dibanding indeks.")
df = add_score(df, PRESETS[preset]["w"])
m = (df.Close >= f_price) & (df.Close <= f_pmax) & (df.rs >= f_rs) & (df.val >= f_val) & (df.rvol >= f_rvol) & (df.chg >= f_chg) \
    & (df.chg <= f_chg_max) & (df.pos >= f_pos) & (df.near <= f_near) & (df.atr_pct >= f_atr)
if f_trend:
    m &= df.trend
out = df[m].sort_values("Skor", ascending=False).copy()

st.subheader(f"{len(out)} dari {len(df)} saham lolos")
if out.empty:
    st.warning("Tidak ada yang lolos. Longgarkan filter di sidebar atau coba preset lain.")
else:
    plan = make_plan(out, modal, risk_pct, alloc_pct)
    out = out.join(plan)
    show = out[["Kode", "Skor", "Close", "chg", "rs", "rvol", "pos", "near", "atr_pct",
                "Trigger", "Stop", "Target", "Risiko", "Batal", "Lot", "Nilai"]].rename(columns={
        "chg": "Ubah %", "rs": "vs IHSG", "rvol": "Vol rel", "pos": "Posisi close", "near": "Jarak high20 %",
        "atr_pct": "ATR %", "Risiko": "Risiko %", "Batal": "Batal jika buka >", "Nilai": "Nilai beli (Rp jt)"})
    st.dataframe(
        show.style.format({"Close": "{:,.0f}", "Ubah %": "{:+.2f}", "vs IHSG": "{:+.2f}", "Vol rel": "{:.2f}x", "Posisi close": "{:.2f}",
                           "Jarak high20 %": "{:.1f}", "ATR %": "{:.1f}", "Trigger": "{:,.0f}", "Stop": "{:,.0f}",
                           "Target": "{:,.0f}", "Risiko %": "{:.1f}", "Batal jika buka >": "{:,.0f}",
                           "Nilai beli (Rp jt)": "{:.1f}"}),
        hide_index=True, use_container_width=True)
    st.download_button("Unduh rencana (CSV)", show.to_csv(index=False).encode(), "rencana-besok.csv", "text/csv")
    if (plan.Lot == 0).any():
        st.caption("Lot 0 berarti jarak ke stop terlalu lebar untuk batas risiko atau alokasi yang kamu set. Lewati saham itu.")

if not out.empty and (out.Close < 100).any():
    st.warning("Saham di bawah Rp100 bergerak besar per fraksi harga (Rp1 pada harga Rp50 = 2%), spread-nya lebar, "
               "dan sering tidak ada pembeli saat kamu mau jual. Cek antrean bid di aplikasi broker sebelum masuk, "
               "dan jangan bawa posisi menginap. Batas turun harian (ARB) akan melebar mulai Januari 2027.")

with st.expander("Cara membaca rencana dan checklist pagi", expanded=False):
    st.markdown("""
**Kolom rencana (dihitung dari bar hari ini):**
- **Trigger**: beli hanya jika harga menembus high hari ini. Tidak tembus berarti tidak ada transaksi.
- **Stop**: low hari ini, tapi jaraknya dibatasi maksimal 1x ATR dari trigger. Pasang segera setelah terbeli.
- **Target**: 2x jarak risiko dari trigger (risk:reward 1:2).
- **Batal jika buka >**: kalau saham dibuka di atas harga ini (gap terlalu tinggi), jangan dikejar. Risikonya jadi tidak sepadan.
- **Lot**: dihitung dari modal dan persen risikomu, dibatasi maksimal alokasi per saham.

**Checklist pagi:**
1. Cek berita atau pengumuman emiten semalam dan kondisi IHSG pembukaan. Kalau IHSG turun tajam, kecilkan ukuran posisi atau lewati; saham yang kuat saat IHSG merah masih bisa jadi kandidat, tapi risikonya lebih tinggi.
2. Lihat harga pembukaan. Di bawah stop atau di atas batas "Batal", coret dari daftar.
3. Pasang order beli hanya di dekat trigger, dan hanya untuk saham yang masih memenuhi rencana.
4. Setelah terbeli, langsung tetapkan stop dan target. Jangan geser stop menjauh.
5. Pilih satu waktu keluar paksa sebelum penutupan, dan jual semuanya di waktu itu walau belum kena target atau stop.
6. Catat hasilnya di jurnal.

Fraksi harga di atas adalah perkiraan; cocokkan dengan aplikasi broker. Pergerakan setelah pembukaan tidak bisa diprediksi, jadi rencana ini bukan jaminan.
""")

if missing:
    st.caption("Data tidak tersedia untuk: " + ", ".join(missing))
