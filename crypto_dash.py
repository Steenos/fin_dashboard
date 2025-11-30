import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto Institutional Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the "Cyber/Dark" aesthetic
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
    }
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stat-value { font-size: 24px; font-weight: bold; color: #ffffff; }
    .stat-label { font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 1px;}
    .stat-sub { font-size: 12px; color: #555; margin-top: 5px; }
    
    /* Neon Text highlights */
    .neon-green { color: #00FF99; font-weight: bold; text-shadow: 0 0 5px rgba(0,255,153,0.5); }
    .neon-red { color: #FF3366; font-weight: bold; text-shadow: 0 0 5px rgba(255,51,102,0.5); }
    .neon-orange { color: #FFCC00; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ Crypto Command")
    st.markdown("---")
    
    # Asset Selection - Hyperliquid Symbols
    asset_map = {
        "Bitcoin (BTC)": "BTC",
        "Ethereum (ETH)": "ETH",
        "Hyperliquid (HYPE)": "HYPE",
        "Solana (SOL)": "SOL",
        "Ripple (XRP)": "XRP",
        "Dogecoin (DOGE)": "DOGE",
        "Cardano (ADA)": "ADA",
        "Binance Coin (BNB)": "BNB",
        "Avalanche (AVAX)": "AVAX"
    }
    
    selected_asset_label = st.selectbox("Select Asset", list(asset_map.keys()), index=0)
    ticker = asset_map[selected_asset_label]
    
    # Benchmark for Correlation (Still uses Yahoo Finance)
    benchmark_map = {
        "Nasdaq 100 (Tech)": "QQQ",
        "S&P 500 (TradFi)": "^GSPC",
        "Gold (Safe Haven)": "GC=F",
        "Dollar Index (DXY)": "DX-Y.NYB"
    }
    benchmark_label = st.selectbox("Correlation Benchmark", list(benchmark_map.keys()), index=0)
    benchmark_ticker = benchmark_map[benchmark_label]
    
    # Timeframe
    # Hyperliquid uses '1d', '1w'. Yahoo uses '1d', '1wk'
    interval_options = ["Daily", "Weekly", "Hourly"]
    timeframe_label = st.selectbox("Timeframe", interval_options, index=0)
    
    # Map for Hyperliquid
    hl_interval_map = {"Daily": "1d", "Weekly": "1w", "Hourly": "1h"}
    hl_interval = hl_interval_map[timeframe_label]
    
    st.markdown("### ⚙️ Engine Settings")
    lookback_period = st.slider("Analysis Horizon (Days)", 90, 730, 365)
    
    st.info(f"Tracking: {ticker} (Hyperliquid) vs {benchmark_ticker} (Yahoo)")

# -----------------------------------------------------------------------------
# 3. Data Fetching & Processing
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300) # Cache for 5 mins
def fetch_hyperliquid_data(coin_symbol, interval, start_time_dt):
    """
    Fetches candle data from Hyperliquid API.
    """
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}
    
    # Convert start datetime to milliseconds timestamp
    start_ts = int(start_time_dt.timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000)
    
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin_symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return pd.DataFrame()
            
        # Parse into DataFrame
        df = pd.DataFrame(data)
        # Hyperliquid returns: t (time), o (open), h (high), l (low), c (close), v (volume)
        # Rename columns to match standard format
        df = df.rename(columns={
            "t": "Date", 
            "o": "Open", 
            "h": "High", 
            "l": "Low", 
            "c": "Close", 
            "v": "Volume"
        })
        
        # Convert types
        df["Date"] = pd.to_datetime(df["Date"], unit="ms")
        df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        
        df = df.set_index("Date")
        return df
        
    except Exception as e:
        st.error(f"Hyperliquid API Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_crypto_data(symbol, bench_symbol, interval, period_days):
    """
    Fetches Crypto (Hyperliquid) and Benchmark (Yahoo) data, calculating 'Hero' metrics.
    """
    start_date = datetime.now() - timedelta(days=period_days + 200) # Buffer for MA
    
    # 1. Fetch Crypto Data from Hyperliquid
    df = fetch_hyperliquid_data(symbol, interval, start_date)
    
    if df.empty:
        return pd.DataFrame()
    
    # 2. Fetch Benchmark Data from Yahoo Finance
    # Map HL interval to YF interval
    yf_interval = interval
    if interval == '1w': yf_interval = '1wk'
    
    try:
        bench = yf.download(bench_symbol, start=start_date, interval=yf_interval, progress=False)
        if isinstance(bench.columns, pd.MultiIndex):
            bench.columns = bench.columns.get_level_values(0)
            
        # Align Benchmark to Crypto Data
        # Ensure index is timezone naive for compatibility if needed, or handle matching
        if bench.index.tz is not None:
             bench.index = bench.index.tz_localize(None)
             
        df['Bench_Close'] = bench['Close'].reindex(df.index).fillna(method='ffill')
    except Exception as e:
        df['Bench_Close'] = df['Close'] # Fallback
        
    # --- PROPRIETARY CALCULATIONS ---
    
    # A. The Macro Engine: Cycle Trend (50D vs 200D MA)
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Trend_Engine'] = (df['MA50'] - df['MA200']) / df['MA200'] * 100
    
    # B. Positioning: RSI (The Crowd)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['Crowd_Sentiment'] = 100 - (100 / (1 + rs))
    
    # C. Volatility: Realized Vol (The Coil)
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(365) * 100
    
    # D. Alignment: Rolling Correlation
    df['Correlation'] = df['Close'].rolling(window=30).corr(df['Bench_Close'])
    
    # Clean up
    df = df.dropna()
    if len(df) > period_days:
        df = df.iloc[-period_days:]
        
    return df

# Fetch Data
with st.spinner('Accessing Hyperliquid Chain Data...'):
    df = fetch_crypto_data(ticker, benchmark_ticker, hl_interval, lookback_period)

if df.empty:
    st.error("Data unavailable. Check if asset exists on Hyperliquid.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. Executive Summary (The Verdict)
# -----------------------------------------------------------------------------
st.title(f"{selected_asset_label} Institutional Dashboard")
st.caption(f"Real-Time Data: Hyperliquid API | Benchmark: {benchmark_label}")

# Current State
last_price = df['Close'].iloc[-1]
last_trend = df['Trend_Engine'].iloc[-1]
last_rsi = df['Crowd_Sentiment'].iloc[-1]
last_vol = df['Volatility'].iloc[-1]
last_corr = df['Correlation'].iloc[-1]

# Logic Engine for "The Verdict"
trend_text = "BULLISH (Golden Cross)" if last_trend > 0 else "BEARISH (Death Cross)"
trend_color = "neon-green" if last_trend > 0 else "neon-red"

if last_vol < 30:
    vol_state = "DEEP COMPRESSION"
    vol_desc = "Volatility is incredibly low. A massive move is imminent."
    vol_color = "neon-orange"
elif last_vol > 80:
    vol_state = "HIGHLY VOLATILE"
    vol_desc = "Price is swinging wildly. Risk management is critical."
    vol_color = "neon-red"
else:
    vol_state = "NORMAL FLOW"
    vol_desc = "Standard market activity."
    vol_color = "white"

if last_rsi > 70:
    crowd_state = "EUPHORIC (Overbought)"
    crowd_desc = "The crowd is all-in. Danger of a pullback/trap."
elif last_rsi < 30:
    crowd_state = "FEARFUL (Oversold)"
    crowd_desc = "Capitulation phase. Value investors are watching."
else:
    crowd_state = "NEUTRAL"
    crowd_desc = "No extreme sentiment reading."

# Display Verdict
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"""
    ### Market Regime: <span class="{trend_color}">{trend_text}</span>
    
    * **The Trend:** {selected_asset_label} is in a **{ 'positive' if last_trend > 0 else 'negative' }** cycle structure (50D vs 200D MA). The engine is pushing {f'{last_trend:.2f}%'}.
    * **The Crowd:** Sentiment is **{crowd_state}** (RSI: {last_rsi:.0f}). {crowd_desc}
    * **The Risk:** Volatility is **{last_vol:.1f}%** ({vol_state}). {vol_desc}
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    **Current Price:**
    # ${last_price:,.2f}
    
    **Alignment:**
    Correlation with {benchmark_label}: **{last_corr:.2f}**
    *{"Decoupled (Idiosyncratic)" if abs(last_corr) < 0.5 else "Highly Correlated (Macro Driven)"}*
    """)

st.divider()

# -----------------------------------------------------------------------------
# 5. KPI Metrics Row
# -----------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

def metric_card(title, value, status, color_cls="white"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="stat-label">{title}</div>
        <div class="stat-value {color_cls}">{value}</div>
        <div class="stat-sub">{status}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi1:
    metric_card("CYCLE TREND", f"{last_trend:.2f}%", "50D-200D Spread", 
                "neon-green" if last_trend > 0 else "neon-red")

with kpi2:
    color = "neon-red" if last_rsi > 70 or last_rsi < 30 else "neon-orange"
    metric_card("CROWD (RSI)", f"{last_rsi:.0f}", crowd_state, color)

with kpi3:
    metric_card("REALIZED VOL", f"{last_vol:.1f}%", vol_state, 
                "neon-red" if last_vol > 80 else "white")

with kpi4:
    metric_card("ALIGNMENT", f"{last_corr:.2f}", f"vs {benchmark_label}", "white")

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. Advanced Charting
# -----------------------------------------------------------------------------
st.subheader("⛓️ On-Chain & Technical Structure")

fig = make_subplots(
    rows=5, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.3, 0.15, 0.15, 0.20, 0.20],
    subplot_titles=(
        f"Price Action: {ticker} (Hyperliquid)", 
        "Cycle Engine (Trend Spread)", 
        f"Alignment (Correlation vs {benchmark_label})", 
        "Crowd Sentiment (RSI)", 
        "Volatility Regime (Realized)"
    )
)

# 1. Price
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], line=dict(color='white', width=1), name="Price"), row=1, col=1)

# 2. Macro Engine (Trend Spread)
# Color the bars based on trend
colors = ['#00FF99' if v > 0 else '#FF3366' for v in df['Trend_Engine']]
fig.add_trace(go.Bar(
    x=df.index, y=df['Trend_Engine'],
    marker=dict(color=colors),
    name="Trend Spread"
), row=2, col=1)

# 3. Alignment
fig.add_trace(go.Scatter(x=df.index, y=df['Correlation'], line=dict(color='#00CCFF', width=1.5), name="Correlation"), row=3, col=1)
fig.add_shape(type="line", x0=df.index[0], y0=0, x1=df.index[-1], y1=0, line=dict(color="gray", width=0.5, dash="dash"), row=3, col=1)

# 4. Crowd (RSI)
fig.add_trace(go.Scatter(x=df.index, y=df['Crowd_Sentiment'], line=dict(color='#FFCC00', width=1.5), name="RSI"), row=4, col=1)
# Overbought/Oversold bands
fig.add_shape(type="line", x0=df.index[0], y0=70, x1=df.index[-1], y1=70, line=dict(color="red", width=0.5, dash="dot"), row=4, col=1)
fig.add_shape(type="line", x0=df.index[0], y0=30, x1=df.index[-1], y1=30, line=dict(color="green", width=0.5, dash="dot"), row=4, col=1)

# 5. Volatility
fig.add_trace(go.Scatter(x=df.index, y=df['Volatility'], line=dict(color='#FF3366', width=1.5), name="Vol"), row=5, col=1)

# Layout
fig.update_layout(
    height=1000,
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=20, r=20, t=40, b=20),
    showlegend=False
)
fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor='#333')
fig.update_xaxes(showgrid=False)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <i>'Cycle Trend' calculated via 50D/200D MA Divergence. Volatility is annualized based on 365 trading days.</i>
</div>
""", unsafe_allow_html=True)