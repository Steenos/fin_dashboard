import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_datareader.data as web
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Kerberos007 Trading Suite V3", layout="wide", page_icon="🐺")

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
    .metric-box-warn {
        background-color: #1E1E1E;
        border-left: 5px solid #FF9800;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    h1, h2, h3 { font-family: 'Roboto', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- DATA FETCHING ---

@st.cache_data(ttl=300)
def fetch_market_data(period="2y"):
    """Fetches Price/Volume data from Yahoo Finance."""
    tickers = {
        "SPX": "^GSPC", 
        "VIX": "^VIX", 
        "VIX3M": "^VIX3M", 
        "XLY": "XLY",   # Discretionary
        "XLP": "XLP",   # Staples
        "RSP": "RSP",   # Equal Weight SP500
        "SPY": "SPY"    # Proxy for Volume
    }
    
    data = {}
    raw_df = yf.download(list(tickers.values()), period=period, group_by='ticker', progress=False)
    
    for key, ticker in tickers.items():
        try:
            df = raw_df[ticker].copy()
            if df.empty: continue
            
            # Cleaning
            df = df.dropna()
            
            # Technicals: Bollinger Bands & %B
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['StdDev'] = df['Close'].rolling(window=20).std()
            df['Upper_BB'] = df['SMA_20'] + (2 * df['StdDev'])
            df['Lower_BB'] = df['SMA_20'] - (2 * df['StdDev'])
            df['Pct_B'] = (df['Close'] - df['Lower_BB']) / (df['Upper_BB'] - df['Lower_BB'])
            
            # Returns
            df['Returns'] = df['Close'].pct_change()
            
            data[key] = df
        except KeyError:
            continue
            
    return data

@st.cache_data(ttl=3600)
def fetch_macro_data():
    """Fetches Official Yield Data from FRED (St. Louis Fed)."""
    start = datetime.datetime.now() - datetime.timedelta(days=365*2)
    end = datetime.datetime.now()
    
    # DGS10: 10-Year Treasury Constant Maturity Rate
    # T10Y2Y: 10-Year Minus 2-Year Treasury Yield Spread
    try:
        macro_df = web.DataReader(['DGS10', 'T10Y2Y'], 'fred', start, end)
        macro_df = macro_df.ffill()
        return macro_df
    except Exception as e:
        st.error(f"Error fetching FRED data: {e}")
        return pd.DataFrame()

# --- LOGIC: KERBEROS NO DEMAND ---
def calculate_no_demand(df):
    """
    Implements Kerberos 'No Demand' Pine Script Logic.
    """
    df = df.copy()
    # 1. Volume SMA
    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    
    # 2. Price Direction
    df['Price_Up'] = df['Close'] > df['Open']
    df['Price_Down'] = df['Close'] < df['Open']
    
    # 3. Previous Highs/Lows
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    
    # 4. Logic Conditions
    # No Demand: Up Candle + Low Volume + Pushing Highs
    df['No_Demand'] = (df['Price_Up']) & \
                      (df['Volume'] < df['Vol_SMA20'] * 0.65) & \
                      (df['Close'] >= df['Prev_High'])
                      
    # High Vol Dump: Down Candle + High Vol + Breaking Lows
    df['High_Vol_Dump'] = (df['Price_Down']) & \
                          (df['Volume'] > df['Vol_SMA20'] * 1.5) & \
                          (df['Close'] <= df['Prev_Low'])
    
    return df

# --- PLOTTING FUNCTIONS ---

def plot_volume_traps(df):
    """
    Plots Candles with specific markers for No Demand / High Vol Dump
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    # Candles
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    
    # --- ANNOTATIONS (The Kerberos Signals) ---
    
    # 1. No Demand (Orange Triangles Down)
    no_demand_dates = df[df['No_Demand']]
    if not no_demand_dates.empty:
        fig.add_trace(go.Scatter(
            x=no_demand_dates.index, y=no_demand_dates['High'] * 1.005,
            mode='markers', marker=dict(symbol='triangle-down', size=12, color='orange'),
            name="No Demand (Trap)"
        ), row=1, col=1)

    # 2. High Vol Dump (Purple Triangles Up)
    dump_dates = df[df['High_Vol_Dump']]
    if not dump_dates.empty:
        fig.add_trace(go.Scatter(
            x=dump_dates.index, y=dump_dates['Low'] * 0.995,
            mode='markers', marker=dict(symbol='triangle-up', size=12, color='#D500F9'),
            name="High Vol Dump"
        ), row=1, col=1)

    # Volume Bar Chart
    colors = ['#FF5252' if c < o else '#4CAF50' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Vol_SMA20'], line=dict(color='yellow', width=1), name="Vol SMA"), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=600, title="Kerberos Volume Trap Detector (SPY)", showlegend=False)
    return fig

def plot_jaws(df_spx, df_yield, title1, title2):
    # Normalize
    norm_spx = (df_spx['Close'] / df_spx['Close'].iloc[0] - 1) * 100
    norm_yield = (df_yield / df_yield.iloc[0] - 1) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_spx.index, y=norm_spx, name=title1, line=dict(color='#00E5FF')))
    fig.add_trace(go.Scatter(x=df_yield.index, y=norm_yield, name=title2, line=dict(color='#FF5252')))
    
    fig.update_layout(template="plotly_dark", height=450, title="The 'Jaws' Divergence (Prices vs Rates)", hovermode="x unified")
    return fig

# --- MAIN APP ---

st.sidebar.title("⚙️ Kerberos Control")
period_select = st.sidebar.select_slider("Lookback", options=["6mo", "1y", "2y", "5y"], value="1y")
vix_panic = st.sidebar.number_input("VIX Panic (%B > X)", 1.0, 1.5, 1.0)

# Fetch Data
with st.spinner("Accessing Institutional Feeds (Yahoo + FRED)..."):
    market_data = fetch_market_data(period_select)
    macro_data = fetch_macro_data()

if not market_data:
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "1. FGMR (Signals)", 
    "2. Volume Traps (New)", 
    "3. Quant Lab (Jaws)", 
    "4. Macro (FRED)"
])

# --- TAB 1: FGMR ---
with tab1:
    st.markdown("### 🛡️ Fear & Greed Mean Reversion")
    vix_df = market_data['VIX']
    curr_b = vix_df['Pct_B'].iloc[-1]
    
    # Term Structure
    common_idx = vix_df.index.intersection(market_data['VIX3M'].index)
    ratio = vix_df.loc[common_idx]['Close'] / market_data['VIX3M'].loc[common_idx]['Close']
    curr_ratio = ratio.iloc[-1]
    
    col1, col2 = st.columns(2)
    with col1:
        if curr_b > vix_panic:
            st.markdown(f"<div class='metric-box'><h3>SIGNAL: BUY (Panic)</h3><p>VIX %B {curr_b:.2f} > 1.0</p></div>", unsafe_allow_html=True)
        elif curr_b < 0:
            st.markdown(f"<div class='metric-box-bear'><h3>SIGNAL: SELL (Greed)</h3><p>VIX %B {curr_b:.2f} < 0.0</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='metric-box-warn'><h3>SIGNAL: NEUTRAL</h3><p>VIX %B {curr_b:.2f}</p></div>", unsafe_allow_html=True)
            
    with col2:
        st.metric("VIX Term Structure", f"{curr_ratio:.2f}", delta="Backwardation > 1.0" if curr_ratio > 1 else "Contango < 1.0", delta_color="inverse")

    # Plot VIX with Bands
    fig_vix = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig_vix.add_trace(go.Scatter(x=vix_df.index, y=vix_df['Close'], name="VIX", line=dict(color='white')), row=1, col=1)
    fig_vix.add_trace(go.Scatter(x=vix_df.index, y=vix_df['Upper_BB'], line=dict(dash='dot', color='gray'), name="Upper"), row=1, col=1)
    fig_vix.add_trace(go.Scatter(x=vix_df.index, y=vix_df['Lower_BB'], line=dict(dash='dot', color='gray'), name="Lower"), row=1, col=1)
    fig_vix.add_trace(go.Scatter(x=vix_df.index, y=vix_df['Pct_B'], line=dict(color='#00E5FF'), name="%B"), row=2, col=1)
    fig_vix.add_hline(y=1.0, row=2, col=1, line_color="red")
    fig_vix.add_hline(y=0.0, row=2, col=1, line_color="green")
    fig_vix.update_layout(template="plotly_dark", height=500, showlegend=False)
    st.plotly_chart(fig_vix, use_container_width=True)

# --- TAB 2: VOLUME TRAPS (NEW) ---
with tab2:
    st.markdown("### 🕵️ Kerberos 'No Demand' Volume Detector")
    st.caption("Logic: Green Candle + Low Volume + New Highs = **Rug Pull Warning** (Orange). Red Candle + High Volume + New Lows = **Dump** (Purple).")
    
    # Process SPY data for this indicator
    spy_processed = calculate_no_demand(market_data['SPY'])
    
    # Check most recent signal
    last_row = spy_processed.iloc[-1]
    if last_row['No_Demand']:
        st.error("🚨 ALERT: NO DEMAND SIGNAL DETECTED TODAY (Potential Bull Trap)")
    elif last_row['High_Vol_Dump']:
        st.info("📉 ALERT: HIGH VOLUME DUMP DETECTED (Capitulation?)")
    else:
        st.success("No immediate volume anomalies detected today.")
        
    st.plotly_chart(plot_volume_traps(spy_processed), use_container_width=True)

# --- TAB 3: QUANT LAB ---
with tab3:
    st.subheader("The 'Jaws' of Death (SPX vs 10Y Yield)")
    st.markdown("If SPX (Blue) is rising while Yields (Red) are rising, the correlation is breaking. Historically, yields up = stocks down.")
    
    if not macro_data.empty:
        # Sync dates
        df_spx = market_data['SPX']
        df_yield = macro_data['DGS10']
        common = df_spx.index.intersection(df_yield.index)
        
        st.plotly_chart(plot_jaws(df_spx.loc[common], df_yield.loc[common], "SPX", "10Y Yield"), use_container_width=True)
    else:
        st.warning("Macro data unavailable. Check FRED connection.")

# --- TAB 4: MACRO DATA ---
with tab4:
    st.subheader("Official Fed Data (FRED)")
    if not macro_data.empty:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            curr_10y = macro_data['DGS10'].iloc[-1]
            st.metric("10-Year Treasury Yield", f"{curr_10y:.2f}%")
        with col_m2:
            curr_curve = macro_data['T10Y2Y'].iloc[-1]
            st.metric("10Y-2Y Yield Curve", f"{curr_curve:.2f}%", delta="Recession Warning" if curr_curve < 0 else "Normal", delta_color="inverse")
        
        st.line_chart(macro_data)
    else:
        st.write("Data could not be loaded.")