import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FX Institutional Dashboard | Real Data",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stat-value { font-size: 24px; font-weight: bold; color: #ffffff; }
    .stat-label { font-size: 14px; color: #888; }
    .stat-sub { font-size: 12px; color: #555; }
    
    /* Text highlights */
    .highlight-green { color: #00FF7F; font-weight: bold; }
    .highlight-red { color: #FF4B4B; font-weight: bold; }
    .highlight-orange { color: #FFA500; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("FX Hero Control")
    st.markdown("---")
    
    # Asset Selection with Yahoo Finance Ticker Mapping
    asset_map = {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "AUD/USD": "AUDUSD=X",
        "Gold": "GC=F",
        "S&P 500": "^GSPC"
    }
    
    selected_asset_label = st.selectbox("Asset Class", list(asset_map.keys()), index=0)
    ticker = asset_map[selected_asset_label]
    
    # Timeframe selection (mapped to yf interval)
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Hourly": "1h"}
    timeframe_label = st.selectbox("Timeframe", list(interval_map.keys()), index=0)
    yf_interval = interval_map[timeframe_label]
    
    st.markdown("### Model Settings")
    lookback_period = st.slider("Lookback Period (Days)", 60, 730, 365)
    vol_window = st.slider("Vol Smoothing (Days)", 5, 50, 20)
    
    st.success(f"Fetching real data for: {ticker}")

# -----------------------------------------------------------------------------
# 3. Real Data Fetching & Processing
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Cache data for 1 hour to prevent API limits
def fetch_real_data(symbol, interval, period_days):
    """
    Fetches real market data using yfinance and calculates institutional metrics.
    """
    # Calculate start date (buffer added for Moving Averages)
    start_date = datetime.now() - timedelta(days=period_days + 300)
    
    # Fetch Target Asset
    try:
        df = yf.download(symbol, start=start_date, interval=interval, progress=False)
    except Exception as e:
        return pd.DataFrame()
    
    if df.empty:
        return pd.DataFrame()
    
    # Handle MultiIndex columns (yfinance structure update)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except:
            pass
            
    # Ensure Close is float
    df['Close'] = df['Close'].astype(float)
    
    # Fetch Benchmark for Correlation (S&P 500)
    try:
        benchmark = yf.download("^GSPC", start=start_date, interval=interval, progress=False)
        if isinstance(benchmark.columns, pd.MultiIndex):
            benchmark.columns = benchmark.columns.get_level_values(0)
        
        # Merge benchmark close for correlation calculation
        # We reindex to match the asset's dates
        df['Benchmark_Close'] = benchmark['Close'].reindex(df.index).fillna(method='ffill')
    except:
        # Fallback if benchmark fails
        df['Benchmark_Close'] = df['Close']
    
    # --- CALCULATIONS ---
    
    # 1. Price (already there)
    
    # 2. Macro Spread Proxy (Institutional Trend)
    # Using 50-day vs 200-day Moving Average Spread to represent "Trend Strength"
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    # Spread as a percentage of the 200MA
    df['Macro_Spread'] = (df['MA50'] - df['MA200']) / df['MA200'] * 100
    
    # 3. Correlation (Alignment)
    # Rolling correlation between Asset and S&P500 (Risk Sentiment)
    df['Correlation'] = df['Close'].rolling(window=30).corr(df['Benchmark_Close'])
    
    # 4. Positioning Proxy (RSI instead of COT)
    # Since we don't have real COT, we use RSI as a proxy for "Crowdedness"
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 5. Volatility (Realized Annualized)
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
    
    # Drop NaN values created by rolling windows
    df = df.dropna()
    
    # Trim to requested period
    if len(df) > period_days:
        df = df.iloc[-period_days:]
    
    return df

# Fetch Data
try:
    with st.spinner('Loading real market data...'):
        df = fetch_real_data(ticker, yf_interval, lookback_period)
        
    if df.empty:
        st.error(f"No data found for {ticker}. Try a different asset or timeframe. (Note: Yahoo Finance may rate limit frequent requests)")
        st.stop()
        
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 4. Header & Verdict (Dynamic)
# -----------------------------------------------------------------------------
st.title(f"Institutional Dashboard: {selected_asset_label}")
st.caption(f"Real-Time Data Source: Yahoo Finance ({ticker})")

# Determine dynamic status based on real data
current_price = df['Close'].iloc[-1]
current_vol = df['Volatility'].iloc[-1]
current_rsi = df['RSI'].iloc[-1]
current_corr = df['Correlation'].iloc[-1]

# Dynamic Regime Logic
if current_vol < 5:
    regime = "COMPRESSION (Breakout Imminent)"
    regime_color = "orange"
elif current_vol > 12:
    regime = "HIGH VOLATILITY"
    regime_color = "red"
else:
    regime = "NORMAL FLOW"
    regime_color = "#00FF7F"

st.markdown(f"### Market Regime: :{regime_color}[{regime}]")

col_verdict_1, col_verdict_2 = st.columns([2, 1])

with col_verdict_1:
    st.info(f"""
    **Live Analysis:**
    * **Price Action:** Trading at **{current_price:.4f}**.
    * **Positioning (RSI):** currently at **{current_rsi:.1f}**. { 'Overbought!' if current_rsi > 70 else ('Oversold!' if current_rsi < 30 else 'Neutral.') }
    * **Volatility:** Realized vol is **{current_vol:.2f}%**.
    """)

with col_verdict_2:
    # Simple logic for scenario watch based on trend
    trend_dir = "Bullish" if df['Macro_Spread'].iloc[-1] > 0 else "Bearish"
    st.markdown(f"""
    **Structure Watch:**
    * **Trend:** {trend_dir} (50/200 Spread)
    * **Risk:** {'Elevated' if current_vol > 10 else 'Muted'}
    """)

# -----------------------------------------------------------------------------
# 5. KPI Metrics
# -----------------------------------------------------------------------------
def metric_card(title, value, status, color_code="#ffffff"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="stat-label">{title}</div>
        <div class="stat-value" style="color: {color_code}">{value}</div>
        <div class="stat-sub" style="font-weight:bold; margin-top:5px;">{status}</div>
    </div>
    """, unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    trend_val = df['Macro_Spread'].iloc[-1]
    trend_status = "BULLISH TREND" if trend_val > 0 else "BEARISH TREND"
    metric_card("MACRO TREND (50-200 Spread)", f"{trend_val:.2f}%", trend_status, "#00FF7F" if trend_val > 0 else "#FF4B4B")

with kpi2:
    rsi_status = "CROWDED LONG" if current_rsi > 70 else ("CROWDED SHORT" if current_rsi < 30 else "NEUTRAL")
    metric_card("POSITIONING (RSI)", f"{current_rsi:.0f}", rsi_status, "#FFA500")

with kpi3:
    vol_status = "COMPRESSED" if current_vol < 6 else "EXPANDED"
    metric_card("REALIZED VOL (20d)", f"{current_vol:.2f}%", vol_status, "#FF4B4B" if current_vol < 6 else "#FFFFFF")

with kpi4:
    corr_status = "RISK ON" if current_corr > 0.5 else ("RISK OFF" if current_corr < -0.5 else "DECOUPLED")
    metric_card("CORRELATION (vs SPX)", f"{current_corr:.2f}", corr_status, "#888888")

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. Technical Charts
# -----------------------------------------------------------------------------
st.subheader("Technical & Macro Structure")

fig = make_subplots(
    rows=5, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.3, 0.15, 0.15, 0.20, 0.20],
    subplot_titles=(
        "Price Action", 
        "Macro Trend (50d vs 200d MA Spread)", 
        "Correlation: Asset vs S&P 500", 
        "Positioning Proxy (RSI)", 
        "Realized Volatility (Annualized)"
    )
)

# 1. Price
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], line=dict(color='white', width=1), name=selected_asset_label), row=1, col=1)

# 2. Macro Spread (Trend)
colors = ['#00FF7F' if v > 0 else '#FF4B4B' for v in df['Macro_Spread']]
fig.add_trace(go.Bar(
    x=df.index, y=df['Macro_Spread'],
    marker=dict(color=colors),
    name="Trend Spread"
), row=2, col=1)

# 3. Correlation
fig.add_trace(go.Scatter(x=df.index, y=df['Correlation'], line=dict(color='#3399FF', width=1.5), name="Corr vs SPX"), row=3, col=1)
fig.add_shape(type="line", x0=df.index[0], y0=0, x1=df.index[-1], y1=0, line=dict(color="white", width=0.5, dash="dash"), row=3, col=1)

# 4. Positioning (RSI)
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#FFA500', width=1.5), name="RSI"), row=4, col=1)
fig.add_shape(type="line", x0=df.index[0], y0=70, x1=df.index[-1], y1=70, line=dict(color="red", width=0.5, dash="dot"), row=4, col=1)
fig.add_shape(type="line", x0=df.index[0], y0=30, x1=df.index[-1], y1=30, line=dict(color="green", width=0.5, dash="dot"), row=4, col=1)

# 5. Volatility
fig.add_trace(go.Scatter(x=df.index, y=df['Volatility'], line=dict(color='#E066FF', width=1.5), name="Vol"), row=5, col=1)

# Layout
fig.update_layout(height=1000, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor='#333')
fig.update_xaxes(showgrid=False)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <i>Data provided by Yahoo Finance. 'Macro Trend' is approximated via Moving Average spreads. 'Positioning' uses RSI proxy.</i>
</div>
""", unsafe_allow_html=True)