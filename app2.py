import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Kerberos007 Trading Suite", layout="wide", page_icon="🐺")

# --- CSS & STYLING ---
st.markdown("""
<style>
    .metric-box {
        background-color: #1E1E1E;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .metric-box-bear {
        background-color: #1E1E1E;
        border-left: 5px solid #FF5252;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    h1, h2, h3 { font-family: 'Roboto', sans-serif; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA FETCHING & PROCESSING ---
@st.cache_data(ttl=300) # Cache data for 5 minutes
def fetch_and_process_data(period="1y"):
    tickers = {
        "SPX": "^GSPC", 
        "VIX": "^VIX", 
        "VIX3M": "^VIX3M", 
        "TLT": "TLT",   # Bonds
        "XLY": "XLY",   # Discretionary
        "XLP": "XLP",   # Staples
        "RSP": "RSP",   # Equal Weight SP500
        "SPY": "SPY"    # Used for Volume (SPX volume often unreliable)
    }
    
    data = {}
    # Download all at once to save time, then split
    raw_df = yf.download(list(tickers.values()), period=period, group_by='ticker', progress=False)
    
    for key, ticker in tickers.items():
        # Handle yfinance multi-index structure
        try:
            df = raw_df[ticker].copy()
        except KeyError:
            continue
            
        # Basic Cleaning
        df = df.dropna()
        
        # 1. Bollinger Bands & %B
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['StdDev'] = df['Close'].rolling(window=20).std()
        df['Upper_BB'] = df['SMA_20'] + (2 * df['StdDev'])
        df['Lower_BB'] = df['SMA_20'] - (2 * df['StdDev'])
        # %B Formula: (Price - Lower) / (Upper - Lower)
        df['Pct_B'] = (df['Close'] - df['Lower_BB']) / (df['Upper_BB'] - df['Lower_BB'])
        
        # 2. Z-Scores (For Stat Arb)
        # Z = (Price - Mean) / StdDev
        df['Z_Score'] = (df['Close'] - df['SMA_20']) / df['StdDev']
        
        # 3. Returns
        df['Returns'] = df['Close'].pct_change()
        
        data[key] = df
        
    return data

# --- PLOTTING FUNCTIONS ---

def plot_fgmr(df, ticker_name, threshold_high, threshold_low, invert_signal=False):
    """
    Plots Price + Bollinger Bands + %B Oscillator
    invert_signal: True for VIX (High VIX = Buy Signal)
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05)

    # Price & Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='white')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_BB'], name="Upper BB", line=dict(width=1, color='gray', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_BB'], name="Lower BB", line=dict(width=1, color='gray', dash='dot')), row=1, col=1)

    # %B Oscillator
    color = '#00E5FF'
    fig.add_trace(go.Scatter(x=df.index, y=df['Pct_B'], name="%B", line=dict(color=color)), row=2, col=1)
    
    # Thresholds
    fig.add_hline(y=threshold_high, row=2, col=1, line_color="red", line_dash="dash", annotation_text="Greed/Panic")
    fig.add_hline(y=threshold_low, row=2, col=1, line_color="green", line_dash="dash", annotation_text="Value/Complacency")
    fig.add_hline(y=0.5, row=2, col=1, line_color="gray", line_width=0.5)

    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    return fig

def plot_volume_spread(df):
    """
    Plots Price (Candles) + Volume colored by close vs open
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="SPY"), row=1, col=1)
    
    # Volume with MA
    colors = ['#FF5252' if c < o else '#4CAF50' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Vol"), row=2, col=1)
    
    # Vol MA
    vol_ma = df['Volume'].rolling(20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=vol_ma, line=dict(color='yellow', width=1), name="Vol MA"), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=500, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def plot_jaws(df1, df2, name1, name2):
    """
    Normalized comparison for 'Jaws' divergence
    """
    # Normalize to start at 0%
    norm1 = (df1['Close'] / df1['Close'].iloc[0] - 1) * 100
    norm2 = (df2['Close'] / df2['Close'].iloc[0] - 1) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df1.index, y=norm1, name=name1, line=dict(color='#00E5FF')))
    fig.add_trace(go.Scatter(x=df2.index, y=norm2, name=name2, line=dict(color='#FF5252')))
    
    fig.update_layout(template="plotly_dark", height=400, title="Relative Performance (%)", hovermode="x unified")
    return fig

# --- MAIN APP LOGIC ---

st.sidebar.title("⚙️ Kerberos Control")
data_period = st.sidebar.select_slider("Lookback Period", options=["3mo", "6mo", "1y", "2y", "5y"], value="1y")
vix_panic = st.sidebar.number_input("FGMR: VIX Panic Threshold (%B > X)", 1.0, 1.5, 1.0)
vix_complacency = st.sidebar.number_input("FGMR: VIX Complacency (%B < X)", -0.2, 0.2, 0.0)

# Load Data
with st.spinner("Analyzing Institutional Flows..."):
    data = fetch_and_process_data(data_period)

if not data:
    st.error("Failed to load data. Please check your connection.")
    st.stop()

# --- NAVIGATION ---
tab_fgmr, tab_vsa, tab_quant = st.tabs(["1. FGMR & Mean Reversion", "2. Wyckoff & VSA", "3. Quant Lab (Stat Arb)"])

# ==========================================
# TAB 1: FGMR (Fear & Greed Mean Reversion)
# ==========================================
with tab_fgmr:
    st.markdown("### 🛡️ Fear & Greed Mean Reversion (FGMR)")
    st.caption("Core Philosophy: Fade extremes. Buy when VIX breaks upper band (Panic). Sell when VIX breaks lower band (Complacency).")
    
    # VIX Logic
    vix_current_b = data['VIX']['Pct_B'].iloc[-1]
    vix_close = data['VIX']['Close'].iloc[-1]
    
    # Term Structure Logic (Contango vs Backwardation)
    # Align data
    common_idx = data['VIX'].index.intersection(data['VIX3M'].index)
    vix_ratio = data['VIX'].loc[common_idx]['Close'] / data['VIX3M'].loc[common_idx]['Close']
    current_ratio = vix_ratio.iloc[-1]
    
    # Signal Generation
    if vix_current_b > vix_panic:
        signal = "🚨 BUY THE DIP (Extreme Fear)"
        style = "metric-box"
        desc = "VIX is piercing the Upper Bollinger Band. Statistical probability of a bounce is High."
    elif vix_current_b < vix_complacency:
        signal = "⚠️ SHORT THE RIP (Extreme Greed)"
        style = "metric-box-bear"
        desc = "VIX is crushed into the Lower Bollinger Band. Volatility expansion (drop) is imminent."
    else:
        signal = "⚖️ NEUTRAL / CHOP"
        style = "metric-box" # Neutral styling
        desc = "VIX is inside the bands. No statistical edge on volatility extremes."

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"<div class='{style}'><h3>Signal</h3><h2>{signal}</h2><p>{desc}</p></div>", unsafe_allow_html=True)
        
    with col2:
        st.metric("VIX %B", f"{vix_current_b:.2f}", delta="> 1.0 is Panic", delta_color="inverse")
        st.metric("VIX Level", f"{vix_close:.2f}")

    with col3:
        ratio_color = "inverse" if current_ratio < 1 else "normal"
        ratio_state = "Backwardation (Panic)" if current_ratio > 1 else "Contango (Normal)"
        st.metric("VIX Term Structure", f"{current_ratio:.2f}", delta=ratio_state, delta_color=ratio_color)
        st.caption("VIX / VIX3M Ratio. > 1.0 means 'Blood in the streets'.")

    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("VIX: The Signal Generator")
        st.plotly_chart(plot_fgmr(data['VIX'], "VIX", vix_panic, vix_complacency), use_container_width=True)
    with c2:
        st.subheader("SPX: The Tradable Asset")
        st.plotly_chart(plot_fgmr(data['SPX'], "SPX", 1.0, 0.0), use_container_width=True)

# ==========================================
# TAB 2: WYCKOFF & VSA
# ==========================================
with tab_vsa:
    st.markdown("### 📊 Volume Spread Analysis (VSA) & Structure")
    st.caption("Philosophy: 'Effort vs Result'. High volume with no price progress = Reversal. Low volume rallies = Traps.")
    
    # Logic: Compare Volume to MA
    spy_df = data['SPY'] # Using SPY for volume data
    curr_vol = spy_df['Volume'].iloc[-1]
    avg_vol = spy_df['Volume'].rolling(20).mean().iloc[-1]
    vol_ratio = curr_vol / avg_vol
    
    # Breadth Logic (RSP vs SPY)
    # RSP (Equal Weight) / SPY (Cap Weight)
    # If Ratio goes DOWN, it means Mega Caps (Mag 7) are propping up the market (Bad Breadth).
    breadth_ratio = data['RSP']['Close'] / data['SPY']['Close']
    breadth_change = breadth_ratio.pct_change(periods=20).iloc[-1] * 100
    
    col_v1, col_v2, col_v3 = st.columns(3)
    
    with col_v1:
        st.metric("Volume Anomaly", f"{vol_ratio:.2f}x", delta="> 1.5x is High Effort", delta_color="off")
        if vol_ratio < 0.8 and spy_df['Close'].iloc[-1] > spy_df['Open'].iloc[-1]:
             st.warning("⚠️ Low Volume Rally (Potential Bull Trap)")
    
    with col_v2:
        st.metric("Breadth Momentum (20d)", f"{breadth_change:.2f}%", help="Positive = Broad participation. Negative = Narrow leadership.")
        if breadth_change < -2.0:
            st.error("⚠️ Divergence: RSP underperforming SPY (Narrow Breadth)")
            
    with col_v3:
        # Simple Price Position relative to Range
        high = spy_df['High'].iloc[-1]
        low = spy_df['Low'].iloc[-1]
        close = spy_df['Close'].iloc[-1]
        pos = (close - low) / (high - low)
        st.metric("Intraday Close Loc", f"{pos:.2f}", help="Close near 1.0 = Strong, near 0.0 = Weak")

    st.divider()
    
    st.subheader("SPY: Volume & Spread Analysis")
    st.plotly_chart(plot_volume_spread(spy_df), use_container_width=True)
    
    st.subheader("Market Breadth: RSP (Equal) vs SPY (Cap)")
    st.plotly_chart(plot_jaws(data['RSP'], data['SPY'], "RSP (Equal Wt)", "SPY (Cap Wt)"), use_container_width=True)

# ==========================================
# TAB 3: QUANT LAB
# ==========================================
with tab_quant:
    st.markdown("### 🧮 Stat Arb & Macro Correlations")
    st.caption("Philosophy: 'Confluence is King'. Look for divergences in correlated assets.")
    
    q1, q2 = st.columns(2)
    
    with q1:
        st.subheader("The 'Jaws' of Death (SPX vs Bonds)")
        st.markdown("Kerberos often cites the divergence between **Stocks (SPX)** and **Bonds (TLT)**. If SPX makes higher highs while TLT makes lower lows, liquidity is leaving.")
        st.plotly_chart(plot_jaws(data['SPX'], data['TLT'], "SPX (Stocks)", "TLT (Long Bonds)"), use_container_width=True)
        
    with q2:
        st.subheader("Consumer Ratio (Risk On/Off)")
        st.markdown("Ratio of **Discretionary (XLY)** to **Staples (XLP)**. Rising ratio = Risk On. Falling ratio = Defensive Rotation.")
        
        # Calculate Ratio
        xly_xlp = data['XLY']['Close'] / data['XLP']['Close']
        
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(x=xly_xlp.index, y=xly_xlp, name="XLY/XLP", fill='tozeroy', line=dict(color='#AB47BC')))
        fig_ratio.update_layout(template="plotly_dark", height=400, title="XLY / XLP Ratio")
        st.plotly_chart(fig_ratio, use_container_width=True)

    st.divider()
    
    st.subheader("Correlation Matrix (Rolling 20-Day)")
    st.caption("Monitor if correlations break. E.g., if SPX and VIX both go UP (Correlation > 0), this is a major tail-risk signal.")
    
    # Calculate Rolling Correlation of Returns
    # We align the returns first
    combined_rets = pd.DataFrame({
        'SPX': data['SPX']['Returns'],
        'VIX': data['VIX']['Returns'],
        'TLT': data['TLT']['Returns'],
        'Dollar': data['XLP']['Returns'] # Using XLP as proxy for defensive flow for now
    }).dropna()
    
    corr_matrix = combined_rets.tail(20).corr()
    
    # Heatmap
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    fig_corr.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    spx_vix_corr = corr_matrix.loc['SPX', 'VIX']
    if spx_vix_corr > -0.3: # Normally it should be -0.7 or -0.8
        st.warning(f"🚨 SPX-VIX Correlation is Breaking Down: {spx_vix_corr:.2f} (Normal is < -0.7). Tail risk is elevated.")