# MarketMind AI — Portfolio Intelligence (test-now, scale-later)
# Educational only. Not investment advice.

import os, io, json, textwrap, smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from dateutil import tz

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ------------- BRAND / CONFIG -------------
APP_NAME = "MarketMind AI"
TZ = tz.gettz("America/Los_Angeles")
st.set_page_config(page_title=f"{APP_NAME}", layout="wide", page_icon="📊")

# Optional secrets for AI + notifications
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587")) if os.getenv("SMTP_PORT") else None
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
ALERT_TO_EMAIL = os.getenv("ALERT_TO_EMAIL", "")
TWILIO_SID   = os.getenv("TWILIO_SID", "")
TWILIO_AUTH  = os.getenv("TWILIO_AUTH", "")
TWILIO_FROM  = os.getenv("TWILIO_FROM", "")
ALERT_TO_SMS = os.getenv("ALERT_TO_SMS", "")

# OpenAI client (optional)
try:
    from openai import OpenAI
    llm = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    llm = None

sid = SentimentIntensityAnalyzer()

# ------------- STYLE: finance, auto theme -------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
:root { color-scheme: light dark; }
@media (prefers-color-scheme: light) {
  .stApp { background: #ffffff; }
}
@media (prefers-color-scheme: dark) {
  .stApp { background: linear-gradient(180deg,#0B0D12 0%,#0E1117 100%); }
}
.mm-card { background: rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:16px; }
.mm-kpi  { font-weight:600; font-size:24px; }
.mm-kpi-sub { color:#97A6B2; font-size:12px; }
hr { border: 1px solid #1f232b; }
</style>
""", unsafe_allow_html=True)

# ------------- CONSTANTS -------------
INDEX = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "VIX": "^VIX",
    "US Dollar (DXY)": "DX-Y.NYB",
    "10Y Yield": "^TNX",
    "Crude Oil": "CL=F",
    "Gold": "GC=F",
    "Bitcoin": "BTC-USD",
}

DEFAULT_STOCKS = "AAPL, NVDA, MSFT, AMZN, TSLA"
DEFAULT_CRYPTO  = "BTC-USD, ETH-USD, SOL-USD"
DEFAULT_TOPICS  = "Federal Reserve, inflation, earnings, geopolitics, US dollar"

# ------------- HELPERS -------------
def human_dt(dt=None):
    dt = dt or datetime.now(TZ)
    return dt.strftime("%b %d, %Y %I:%M %p %Z")

@st.cache_data(ttl=60*15, show_spinner=False)
def load_price(symbol, period="1y", interval="1d"):
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty: return pd.DataFrame()
    df.index = df.index.tz_localize(TZ) if df.index.tz is None else df.index.tz_convert(TZ)
    df["Returns"] = df["Close"].pct_change()
    return df

def pct_change(df, days):
    if df.empty or len(df)<days+1: return None
    a = df["Close"].iloc[-1]; b = df["Close"].iloc[-1-days]
    return (a/b - 1)*100 if b else None

def google_news_rss(q):
    import urllib.parse as up
    return f"https://news.google.com/rss/search?q={up.quote(q)}&hl=en-US&gl=US&ceid=US:en"

def fetch_news(query, k=10):
    try:
        feed = feedparser.parse(google_news_rss(query))
        items = []
        for e in feed.entries[:k]:
            items.append({"title": e.get("title",""), "link": e.get("link",""), "published": e.get("published","")})
        return items
    except Exception:
        return []

def summarize_headlines(topic, headlines):
    titles = [h["title"] for h in headlines if h.get("title")]
    if llm and titles:
        try:
            r = llm.chat.completions.create(
                model="gpt-4o-mini", temperature=0.2,
                messages=[{"role":"system","content":"Summarize headlines into 2-4 bullets with market implications. No advice."},
                          {"role":"user","content": "\n".join(f"- {t}" for t in titles[:15])}]
            )
            return r.choices[0].message.content.strip()
        except Exception:
            pass
    # fallback
    if titles:
        tone = np.mean([sid.polarity_scores(t)["compound"] for t in titles])
        tilt = "mixed" if abs(tone)<0.1 else ("cautious" if tone<0 else "constructive")
        return f"• **{topic}**: {len(titles)} headlines; tone **{tilt}**. Notable: “{titles[0]}”."
    return f"• **{topic}**: no recent headlines."

def explain_ta():
    st.markdown("""
**How to read the signals**
- **Moving Averages (20/50/200-day)**: trend lines. Above 200-day = longer-term uptrend bias; below = caution.
- **RSI (14)**: momentum 0-100. >70 often “overbought”; <30 “oversold”. Not a guarantee, just a risk zone.
- **MACD**: momentum crossover. MACD above Signal → bullish momentum bias; below → bearish.
    """)

def compute_ta(df):
    if df.empty: return df
    out = df.copy()
    out["SMA_20"] = ta.sma(out["Close"], 20)
    out["SMA_50"] = ta.sma(out["Close"], 50)
    out["SMA_200"] = ta.sma(out["Close"], 200)
    macd = ta.macd(out["Close"], 12, 26, 9)
    if macd is not None and not macd.empty:
        out["MACD"] = macd["MACD_12_26_9"]
        out["MACD_Signal"] = macd["MACDs_12_26_9"]
        out["MACD_Hist"] = macd["MACDh_12_26_9"]
    out["RSI_14"] = ta.rsi(out["Close"], 14)
    return out

def ta_quick_bias(out):
    if out.empty: return "No data."
    L = out.iloc[-1]; parts=[]
    if pd.notnull(L.get("SMA_50")) and pd.notnull(L.get("SMA_200")):
        parts.append("50D>200D (uptrend)" if L["SMA_50"]>L["SMA_200"] else "50D<200D (downtrend)")
    if pd.notnull(L.get("RSI_14")):
        r=L["RSI_14"]
        parts.append(f"RSI {r:.0f} overbought" if r>70 else (f"RSI {r:.0f} oversold" if r<30 else f"RSI {r:.0f} neutral"))
    if pd.notnull(L.get("MACD")) and pd.notnull(L.get("MACD_Signal")):
        parts.append("MACD>Signal (bullish)" if L["MACD"]>L["MACD_Signal"] else "MACD<Signal (bearish)")
    return "; ".join(parts) if parts else "No strong bias."

def send_email(subject, body):
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS and ALERT_TO_EMAIL):
        return False, "Email settings not configured."
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"]=subject; msg["From"]=SMTP_USER; msg["To"]=ALERT_TO_EMAIL
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls(); s.login(SMTP_USER, SMTP_PASS); s.sendmail(SMTP_USER, [ALERT_TO_EMAIL], msg.as_string())
        return True, "Email sent."
    except Exception as e:
        return False, f"Email error: {e}"

def send_sms(body):
    if not (TWILIO_SID and TWILIO_AUTH and TWILIO_FROM and ALERT_TO_SMS):
        return False, "SMS settings not configured."
    try:
        import requests
        url=f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json"
        r=requests.post(url, data={"From":TWILIO_FROM,"To":ALERT_TO_SMS,"Body":body}, auth=(TWILIO_SID,TWILIO_AUTH))
        return (r.status_code in (200,201), ("SMS sent." if r.status_code in (200,201) else r.text))
    except Exception as e:
        return False, f"SMS error: {e}"

def render_pdf(summary_text, details_text, df_table):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W,H = letter; x=40; y=H-50
    def draw(txt, x, y, w=520, leading=12):
        for para in txt.split("\n"):
            lines = textwrap.wrap(para, width=92) or [""]
            for line in lines:
                c.drawString(x,y,line); y-=leading
            y-=6
        return y
    c.setFont("Helvetica-Bold",14); c.drawString(x,y,f"{APP_NAME} — Daily Brief"); y-=18
    c.setFont("Helvetica",10); c.drawString(x,y,human_dt()); y-=14
    c.setFont("Helvetica-Bold",12); c.drawString(x,y,"Summary"); y-=14
    c.setFont("Helvetica",10); y=draw(summary_text,x,y)
    c.setFont("Helvetica-Bold",12); c.drawString(x,y,"Details"); y-=14
    c.setFont("Helvetica",10); y=draw(details_text,x,y)
    c.setFont("Helvetica-Bold",12); c.drawString(x,y,"Portfolio Snapshot"); y-=14
    c.setFont("Helvetica",10)
    if not df_table.empty:
        cols=[c for c in ["Portfolio","Ticker","Qty","Last","Day %","Value","Weight %"] if c in df_table.columns]
        for _,r in df_table[cols].iterrows():
            c.drawString(x,y," | ".join(f"{k}: {r[k]}" for k in cols)); y-=12
            if y<60: c.showPage(); y=H-50; c.setFont("Helvetica",10)
    c.showPage(); c.save(); buf.seek(0); return buf.read()

# ------------- HEADER -------------
st.markdown(f"""
<div style="text-align:center; padding: 8px 0 0;">
  <h1 style="margin:0">{APP_NAME}</h1>
  <p style="color:#97A6B2; margin-top:6px">AI-Powered Portfolio & Market Intelligence</p>
  <hr/>
</div>
""", unsafe_allow_html=True)

# ------------- SIDEBAR INPUTS -------------
with st.sidebar:
    st.subheader("Portfolio Inputs")
    st.caption("Upload one or multiple CSVs (Ticker,Qty,PortfolioName optional).")
    uploads = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
    stocks_text = st.text_area("Stocks (comma-sep)", value=DEFAULT_STOCKS)
    crypto_text = st.text_area("Crypto (comma-sep)", value=DEFAULT_CRYPTO)
    lookback = st.selectbox("Lookback", ["3mo","6mo","1y","2y"], index=2)
    interval = st.selectbox("Interval", ["1d","1h"], index=0)
    st.subheader("Macro Focus")
    topics = st.text_input("World & Macro topics", value=DEFAULT_TOPICS)
    st.subheader("Alert Defaults")
    rsi_low  = st.slider("RSI low", 10, 40, 30)
    rsi_high = st.slider("RSI high", 60, 90, 70)
    big_move = st.slider("1-day move (%)", 1, 10, 4)
    ma200_flag = st.checkbox("Flag if price < 200D MA", True)
    st.caption("Set per-ticker price alerts in Alerts tab.")

# ------------- BUILD HOLDINGS (MULTI-PORTFOLIO) -------------
def parse_uploads(files):
    frames=[]
    for f in files or []:
        try:
            df=pd.read_csv(f)
            df.columns=[c.strip() for c in df.columns]
            if "Ticker" not in df.columns: continue
            if "Qty" not in df.columns: df["Qty"]=1.0
            if "Portfolio" not in df.columns: df["Portfolio"]=os.path.splitext(f.name)[0]
            df["Ticker"]=df["Ticker"].astype(str).str.upper().str.strip()
            frames.append(df[["Portfolio","Ticker","Qty"]])
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Portfolio","Ticker","Qty"])

uploads_df = parse_uploads(uploads)
manual_stocks = [t.strip().upper() for t in (stocks_text or "").split(",") if t.strip()]
manual_crypto = [t.strip().upper() for t in (crypto_text or "").split(",") if t.strip()]

if uploads_df.empty and manual_stocks:
    total_df = pd.DataFrame({"Portfolio":"Manual","Ticker":manual_stocks, "Qty":[1.0]*len(manual_stocks)})
else:
    total_df = uploads_df.copy()
    # Add manual list as another portfolio row if provided
    for t in manual_stocks:
        total_df.loc[len(total_df)] = ["Manual", t, 1.0]

# ------------- DATA LOAD -------------
def price_map_for(tickers):
    return {t: load_price(t, period=lookback, interval=interval) for t in tickers}

all_stock_tickers = sorted(total_df["Ticker"].unique()) if not total_df.empty else []
crypto_tickers = manual_crypto
idx_data = {name: load_price(sym, period="3mo", interval="1d") for name, sym in INDEX.items()}
stock_prices = price_map_for(all_stock_tickers) if all_stock_tickers else {}
crypto_prices = price_map_for(crypto_tickers) if crypto_tickers else {}

# ------------- PORTFOLIO TABLES -------------
def portfolio_table(df, prices):
    rows=[]
    for _,r in df.iterrows():
        port, t, q = r["Portfolio"], r["Ticker"], float(r["Qty"])
        d=prices.get(t, pd.DataFrame())
        last = d["Close"].iloc[-1] if not d.empty else np.nan
        day  = d["Returns"].iloc[-1]*100 if (not d.empty and pd.notnull(d["Returns"].iloc[-1])) else np.nan
        rows.append({"Portfolio":port,"Ticker":t,"Qty":q,
                     "Last": round(float(last),2) if pd.notnull(last) else None,
                     "Day %": round(float(day),2) if pd.notnull(day) else None,
                     "Value": round(float(last*q),2) if pd.notnull(last) else None})
    tab=pd.DataFrame(rows)
    if not tab.empty and tab["Value"].sum():
        tab["Weight %"]=(tab["Value"]/tab["Value"].sum()*100).round(2)
    return tab.sort_values(["Portfolio","Weight %"], ascending=[True,False]) if not tab.empty else tab

stocks_table = portfolio_table(total_df, stock_prices) if not total_df.empty else pd.DataFrame()
crypto_table  = portfolio_table(pd.DataFrame({"Portfolio":"Crypto","Ticker":crypto_tickers,"Qty":[1.0]*len(crypto_tickers)}), crypto_prices) if crypto_tickers else pd.DataFrame()

total_balance = 0.0
if not stocks_table.empty: total_balance += float(stocks_table["Value"].sum())
if not crypto_table.empty: total_balance += float(crypto_table["Value"].sum())

# ------------- TABS -------------
tabs = st.tabs(["🏠 Home","💼 Portfolio","₿ Crypto","🏢 Sectors","🤖 AI Assistant","🔔 Alerts"])

# ========== HOME ==========
with tabs[0]:
    colA, colB = st.columns([1,1])
    with colA:
        st.markdown("### Market Snapshot")
        kcols = st.columns(4)
        card_names = ["S&P 500","Nasdaq 100","Dow Jones","VIX"]
        for i,name in enumerate(card_names):
            with kcols[i]:
                df = idx_data.get(name,pd.DataFrame())
                if df.empty: st.metric(name,"—")
                else:
                    last=float(df["Close"].iloc[-1])
                    day=float(df["Returns"].iloc[-1]*100) if pd.notnull(df["Returns"].iloc[-1]) else 0.0
                    st.metric(name, f"{last:,.2f}", f"{day:+.2f}%")
        st.caption(f"Updated {human_dt()}")

    with colB:
        st.markdown("### Combined Balance")
        st.markdown(f"<div class='mm-card'><div class='mm-kpi'>${total_balance:,.0f}</div><div class='mm-kpi-sub'>All portfolios combined</div></div>", unsafe_allow_html=True)
        if not stocks_table.empty:
            by_port = stocks_table.groupby("Portfolio")["Value"].sum().sort_values(ascending=False)
            st.markdown("**By Portfolio:**")
            for p,v in by_port.items():
                st.markdown(f"- {p}: **${v:,.0f}**")
        if not crypto_table.empty:
            st.markdown(f"- Crypto: **${float(crypto_table['Value'].sum()):,.0f}**")

    st.markdown("---")
    st.markdown("### Why Markets Moved (Today)")
    topics_list = [t.strip() for t in (topics or "").split(",") if t.strip()]
    bullets=[]
    for tpc in topics_list:
        items = fetch_news(tpc, 8)
        bullets.append(summarize_headlines(tpc, items))
    if llm:
        payload = {"indices": {k:(None if v.empty else float(v['Close'].iloc[-1])) for k,v in idx_data.items()},
                   "topics": topics_list, "summaries": bullets}
        try:
            r = llm.chat.completions.create(
                model="gpt-4o-mini", temperature=0.2,
                messages=[{"role":"system","content":"Write 4-7 bullets on why US markets moved today. Neutral, concise. No advice."},
                          {"role":"user","content":json.dumps(payload)[:18000]}]
            )
            st.write(r.choices[0].message.content.strip())
        except Exception as e:
            st.write("\n".join(bullets))
    else:
        st.write("\n".join(bullets))

# ========== PORTFOLIO ==========
with tabs[1]:
    st.markdown("### Portfolio Overview")
    if stocks_table.empty:
        st.info("Upload CSV(s) or add tickers in the sidebar to populate your portfolio.")
    else:
        st.dataframe(stocks_table, use_container_width=True)

        st.markdown("#### Per-Ticker Details")
        explain_ta()
        for t in stocks_table["Ticker"].unique():
            with st.expander(f"{t} — overview"):
                df = stock_prices.get(t, pd.DataFrame())
                if df.empty:
                    st.write("No data."); continue
                # Performance windows
                p1d = (df["Returns"].iloc[-1]*100) if pd.notnull(df["Returns"].iloc[-1]) else None
                p1m = pct_change(df, 21)
                p1y = pct_change(df, 252)
                col1,col2,col3 = st.columns(3)
                col1.metric("1-Day", f"{p1d:+.2f}%" if p1d is not None else "—")
                col2.metric("1-Month", f"{p1m:+.2f}%" if p1m is not None else "—")
                col3.metric("1-Year", f"{p1y:+.2f}%" if p1y is not None else "—")

                # TA
                ta_df = compute_ta(df)
                st.markdown(f"**Technical Snapshot:** {ta_quick_bias(ta_df)}")

                # Charts tucked: 
                with st.expander("Show charts"):
                    fig, ax = plt.subplots(figsize=(7,3))
                    for series,label in [(ta_df["Close"],"Close"),(ta_df["SMA_20"],"SMA20"),(ta_df["SMA_50"],"SMA50"),(ta_df["SMA_200"],"SMA200")]:
                        if series is not None: ax.plot(ta_df.index, series, label=label)
                    ax.legend(loc="upper left", ncols=2); ax.set_title(f"{t} — Price & MAs"); st.pyplot(fig)

                    if "RSI_14" in ta_df.columns:
                        fig, ax = plt.subplots(figsize=(7,2.2))
                        ax.plot(ta_df.index, ta_df["RSI_14"]); ax.axhline(70, ls="--"); ax.axhline(30, ls="--")
                        ax.set_title(f"{t} — RSI"); st.pyplot(fig)
                    if "MACD" in ta_df.columns:
                        fig, ax = plt.subplots(figsize=(7,2.2))
                        ax.plot(ta_df.index, ta_df["MACD"], label="MACD")
                        ax.plot(ta_df.index, ta_df["MACD_Signal"], label="Signal")
                        ax.legend(); ax.set_title(f"{t} — MACD"); st.pyplot(fig)

                # Analyst price targets (best-effort from yfinance, may be missing)
                tgt = None
                try:
                    info = yf.Ticker(t).get_info()
                    if isinstance(info, dict):
                        if "targetMeanPrice" in info and info["targetMeanPrice"]:
                            tgt = float(info["targetMeanPrice"])
                except Exception:
                    pass
                if tgt:
                    last = float(df["Close"].iloc[-1])
                    diff = (tgt/last - 1)*100
                    st.markdown(f"**Analyst Target (avg):** ${tgt:,.2f} ({diff:+.1f}% vs. last)")
                else:
                    st.caption("Analyst targets not available for this symbol via free source.")

                # Stock news summary
                news = fetch_news(t, 8)
                if news:
                    s_text = summarize_headlines(t, news)
                    st.markdown("**Latest News (summary):**")
                    st.write(s_text)
                else:
                    st.caption("No recent headlines found.")

# ========== CRYPTO ==========
with tabs[2]:
    st.markdown("### Crypto")
    if crypto_tickers:
        st.dataframe(crypto_table, use_container_width=True)
        st.markdown("#### Coins")
        for t in crypto_tickers:
            with st.expander(f"{t} — overview"):
                df = crypto_prices.get(t, pd.DataFrame())
                if df.empty: st.write("No data."); continue
                p1d = (df["Returns"].iloc[-1]*100) if pd.notnull(df["Returns"].iloc[-1]) else None
                p1m = pct_change(df, 30)
                p1y = pct_change(df, 365) if len(df)>365 else None
                col1,col2,col3 = st.columns(3)
                col1.metric("1-Day", f"{p1d:+.2f}%" if p1d is not None else "—")
                col2.metric("1-Month", f"{p1m:+.2f}%" if p1m is not None else "—")
                col3.metric("1-Year", f"{p1y:+.2f}%" if p1y is not None else "—")

                with st.expander("Show chart"):
                    fig, ax = plt.subplots(figsize=(7,3))
                    ax.plot(df.index, df["Close"]); ax.set_title(f"{t} — Close")
                    st.pyplot(fig)
    else:
        st.info("Add crypto tickers in the sidebar (e.g., BTC-USD, ETH-USD).")

# ========== SECTORS ==========
with tabs[3]:
    st.markdown("### Sector Overview (approx.)")
    # Approx sector from yfinance info (best-effort)
    sector_rows=[]
    for t in all_stock_tickers:
        try:
            info = yf.Ticker(t).get_info()
            sector = (info.get("sector") or "Other") if isinstance(info, dict) else "Other"
        except Exception:
            sector = "Other"
        v = float(stocks_table.loc[stocks_table["Ticker"]==t,"Value"].sum()) if not stocks_table.empty else 0
        sector_rows.append({"Ticker":t,"Sector":sector,"Value":v})
    s_df = pd.DataFrame(sector_rows)
    if not s_df.empty:
        by_sec = s_df.groupby("Sector")["Value"].sum().sort_values(ascending=False)
        st.dataframe(by_sec.rename("Value ($)").to_frame(), use_container_width=True)
        st.markdown("**Sector Notes**")
        # Headline summaries by sector
        for sec in by_sec.index.tolist():
            with st.expander(f"{sec} — summary"):
                items = fetch_news(f"{sec} stocks", 6)
                st.write(summarize_headlines(sec, items))
    else:
        st.info("No sector data yet (add stocks).")

# ========== AI ASSISTANT ==========
with tabs[4]:
    st.markdown("### Ask MarketMind AI")
    st.caption("Ask about a ticker, macro topic, or your portfolio. Example: “Summarize NVDA news and risks.”")
    q = st.text_input("Your question")
    if st.button("Ask"):
        if llm:
            context = {
                "now": human_dt(),
                "holdings": total_df.to_dict(orient="records"),
                "indices": {k:(None if v.empty else float(v['Close'].iloc[-1])) for k,v in idx_data.items()}
            }
            try:
                r = llm.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.25,
                    messages=[
                        {"role":"system","content":"Be a calm markets analyst. Explain clearly. No investment advice."},
                        {"role":"user","content": f"Context:\n{json.dumps(context)[:15000]}\n\nQuestion:\n{q}"}
                    ]
                )
                st.write(r.choices[0].message.content.strip())
            except Exception as e:
                st.error(f"AI error: {e}")
        else:
            st.info("Add OPENAI_API_KEY in Streamlit secrets to enable AI answers.")

# ========== ALERTS (USER-DEFINED) ==========
with tabs[5]:
    st.markdown("### Price & Event Alerts")
    st.caption("Create custom alerts per symbol. We’ll check live when you load the app; optional email/SMS delivery via secrets.")

    if "alerts" not in st.session_state:
        st.session_state.alerts = []  # list of dicts: {symbol, type, op, value, note}

    with st.form("new_alert"):
        c1,c2,c3,c4 = st.columns([1,1,1,2])
        symbol = c1.text_input("Symbol", value=(all_stock_tickers[0] if all_stock_tickers else "AAPL")).upper().strip()
        atype  = c2.selectbox("Type", ["Price","% Move (1D)"])
        if atype=="Price":
            op = c3.selectbox("When", ["≥ at/above","≤ at/below"])
            val = c4.number_input("Price", min_value=0.0, value=100.0, step=0.1)
        else:
            op = c3.selectbox("When", ["≥","≤"])
            val = c4.number_input("% Move", min_value=0.0, value=float(big_move), step=0.5)
        note = st.text_input("Note (optional)", value="")
        submitted = st.form_submit_button("Add Alert")
        if submitted:
            st.session_state.alerts.append({"symbol":symbol,"type":atype,"op":op,"value":val,"note":note})
            st.success(f"Added alert for {symbol}")

    # Show current alerts
    if st.session_state.alerts:
        st.markdown("#### Your Alerts")
        st.table(pd.DataFrame(st.session_state.alerts))
    else:
        st.info("No alerts yet. Add one above.")

    # Evaluate alerts on current data
    st.markdown("#### Triggered Alerts (now)")
    triggered=[]
    def latest_close(sym):
        mp = stock_prices if sym not in crypto_tickers else crypto_prices
        df = mp.get(sym, pd.DataFrame())
        return float(df["Close"].iloc[-1]) if (not df.empty) else None
    def last_day_move(sym):
        mp = stock_prices if sym not in crypto_tickers else crypto_prices
        df = mp.get(sym, pd.DataFrame())
        if df.empty or pd.isna(df["Returns"].iloc[-1]): return None
        return float(df["Returns"].iloc[-1]*100)

    for al in st.session_state.alerts:
        sym = al["symbol"]
        if al["type"]=="Price":
            px = latest_close(sym)
            if px is None: continue
            if (al["op"].startswith("≥") and px >= al["value"]) or (al["op"].startswith("≤") and px <= al["value"]):
                triggered.append(f"{sym}: price {px:,.2f} {al['op']} {al['value']:,.2f}  {('— '+al['note']) if al['note'] else ''}")
        else:
            mv = last_day_move(sym)
            if mv is None: continue
            if (al["op"]=="≥" and mv >= al["value"]) or (al["op"]=="≤" and mv <= al["value"]):
                triggered.append(f"{sym}: 1-day move {mv:+.2f}% {al['op']} {al['value']:.2f}%  {('— '+al['note']) if al['note'] else ''}")

    if triggered:
        for t in triggered: st.warning("• "+t)
        cA, cB = st.columns(2)
        if cA.button("Email me alerts"):
            ok, msg = send_email("MarketMind Alerts", "\n".join(triggered))
            st.toast(msg, icon="📧" if ok else "⚠️")
        if cB.button("SMS me alerts"):
            ok, msg = send_sms("\n".join(triggered)[:1400])
            st.toast(msg, icon="📱" if ok else "⚠️")
    else:
        st.success("No alerts triggered right now.")

# ------------- FOOTER / EXPORT -------------
st.markdown("---")
colX,colY = st.columns([3,1])
with colX:
    st.caption("Data via Yahoo Finance & Google News RSS. Accuracy not guaranteed. Educational use only — not financial advice.")
with colY:
    if not stocks_table.empty or not crypto_table.empty:
        if st.button("Download PDF Brief"):
            # simple combined summary for PDF
            summary = f"Combined balance: ${total_balance:,.0f}. Updated {human_dt()}."
            details = "This brief includes portfolio tables and alert notes."
            pdf = render_pdf(summary, details, pd.concat([stocks_table, crypto_table], ignore_index=True, sort=False))
            st.download_button("Get PDF", data=pdf, file_name="MarketMind_Daily_Brief.pdf", mime="application/pdf")
