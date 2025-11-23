import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- Configuration & Methodology ---
# Kerberos007 Strategy: FGMR (Fear & Greed Mean Reversion)
# Core logic: Fade VIX extremes. 
# VIX > Upper BB (Panic) = Buy Stocks.
# VIX < Lower BB (Complacency) = Sell Stocks.

st.set_page_config(page_title="Kerberos007 Market Monitor", layout="wide", page_icon="📉")

# --- CSS Styling ---
st.markdown("""
<style>
    .metric-container {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .bullish { color: #00FF00; font-weight: bold; }
    .bearish { color: #FF0000; font-weight: bold; }
    .neutral { color: #888888; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def get_data(ticker, period="1y", interval="1d"):
    """Fetches data and calculates Bollinger Bands & %B."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    # Handle multi-index columns if they exist (yfinance update)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Calculate Bollinger Bands (20, 2 standard logic)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['StdDev'] = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = df['SMA_20'] + (2 * df['StdDev'])
    df['Lower_BB'] = df['SMA_20'] - (2 * df['StdDev'])
    
    # Calculate %B
    # %B = (Price - Lower Band) / (Upper Band - Lower Band)
    df['Pct_B'] = (df['Close'] - df['Lower_BB']) / (df['Upper_BB'] - df['Lower_BB'])
    
    return df

def plot_kerberos_chart(df, ticker, title, signal_type="Standard"):
    """
    Creates a chart with Price, Bollinger Bands, and %B Oscillator.
    signal_type="Inverse" for VIX (High VIX = Bullish Stocks).
    """
    last_close = df['Close'].iloc[-1]
    last_pct_b = df['Pct_B'].iloc[-1]
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3],
                        subplot_titles=(f"{title} Price & Bands", "%B Oscillator (Momentum)"))

    # Candlestick (simplified to Line for clarity with Bands) or actual Candles
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_BB'], line=dict(color='gray', width=1, dash='dot'), name="Upper BB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_BB'], line=dict(color='gray', width=1, dash='dot'), name="Lower BB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1), name="SMA 20"), row=1, col=1)

    # %B Oscillator
    fig.add_trace(go.Scatter(x=df.index, y=df['Pct_B'], line=dict(color='#00CC96', width=2), name="%B"), row=2, col=1)
    
    # %B Threshold Lines
    fig.add_hline(y=1.0, line_dash="dot", row=2, col=1, annotation_text="Overbought (1.0)", annotation_position="top left", line_color="red")
    fig.add_hline(y=0.0, line_dash="dot", row=2, col=1, annotation_text="Oversold (0.0)", annotation_position="bottom left", line_color="green")
    fig.add_hline(y=0.5, line_width=1, line_color="gray", row=2, col=1)

    # Styling
    fig.update_layout(height=600, showlegend=False, template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="%B Value", row=2, col=1)
    
    return fig

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Strategy Settings")
bb_period = st.sidebar.number_input("BB Period", value=20)
bb_std = st.sidebar.number_input("BB Std Dev", value=2.0)
vix_upper_thresh = st.sidebar.slider("VIX Panic Threshold (%B > X)", 0.8, 1.5, 1.0)
vix_lower_thresh = st.sidebar.slider("VIX Complacency Threshold (%B < X)", -0.5, 0.2, 0.0)

# --- Main App ---
st.title("👁️ Kerberos007 Market Monitor")
st.markdown("### FGMR (Fear & Greed Mean Reversion) Dashboard")
st.info("Methodology: Fade VIX extremes. When VIX breaks its Upper Bollinger Band (Panic), BUY SPX. When VIX hits Lower Band (Complacency), SELL/SHORT SPX.")

# Fetch Data
with st.spinner("Fetching market data..."):
    # Tickers: SPX (^GSPC), VIX (^VIX), 10Y Yield (^TNX), Biotech (XBI), Financials (XLF)
    spx = get_data("^GSPC")
    vix = get_data("^VIX")
    xbi = get_data("XBI")
    xlf = get_data("XLF")

# --- Signal Logic ---
current_vix_b = vix['Pct_B'].iloc[-1]
current_spx_b = spx['Pct_B'].iloc[-1]

signal_text = "NEUTRAL"
signal_color = "neutral"
sub_text = "Market is in normal volatility regime."

# Kerberos Logic
if current_vix_b >= vix_upper_thresh:
    signal_text = "🚀 BUY SIGNAL (Max Fear)"
    signal_color = "bullish"
    sub_text = f"VIX %B is {current_vix_b:.2f} (Above {vix_upper_thresh}). Panic is extreme. Expect mean reversion up."
elif current_vix_b <= vix_lower_thresh:
    signal_text = "⚠️ SELL SIGNAL (Complacency)"
    signal_color = "bearish"
    sub_text = f"VIX %B is {current_vix_b:.2f} (Below {vix_lower_thresh}). Market is too calm. Expect volatility spike."

# --- Dashboard Grid ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"<div class='metric-container'><h3>Current Signal</h3><h2 class='{signal_color}'>{signal_text}</h2><p>{sub_text}</p></div>", unsafe_allow_html=True)

with col2:
    v_val = vix['Close'].iloc[-1]
    v_change = (v_val - vix['Close'].iloc[-2]) / vix['Close'].iloc[-2] * 100
    st.metric("VIX Level", f"{v_val:.2f}", f"{v_change:.2f}%")
    st.metric("VIX %B", f"{current_vix_b:.2f}", delta_color="inverse")

with col3:
    s_val = spx['Close'].iloc[-1]
    s_change = (s_val - spx['Close'].iloc[-2]) / spx['Close'].iloc[-2] * 100
    st.metric("SPX Level", f"{s_val:.2f}", f"{s_change:.2f}%")
    st.metric("SPX %B", f"{current_spx_b:.2f}")

st.markdown("---")

# --- Charts ---
tab1, tab2, tab3 = st.tabs(["📊 VIX Analysis (Leading)", "📈 SPX Analysis (Lagging)", "🏥 Sectors (XBI/XLF)"])

with tab1:
    st.subheader("VIX: The Fear Gauge")
    st.plotly_chart(plot_kerberos_chart(vix, "VIX", "Volatility Index"), use_container_width=True)
    st.caption("**How to read:** Look for the Green Line (%B) to cross above 1.0 (Red Dashed Line) for a Buy Signal, or below 0.0 (Green Dashed Line) for a Sell/Caution Signal.")

with tab2:
    st.subheader("S&P 500 Price Action")
    st.plotly_chart(plot_kerberos_chart(spx, "SPX", "S&P 500"), use_container_width=True)

with tab3:
    st.subheader("Risk-On vs Risk-Off Sectors")
    # Normalize comparison
    norm_period = 60 # last 60 days
    df_norm = pd.DataFrame()
    df_norm['SPX'] = spx['Close'].tail(norm_period) / spx['Close'].tail(norm_period).iloc[0]
    df_norm['XBI (Biotech)'] = xbi['Close'].tail(norm_period) / xbi['Close'].tail(norm_period).iloc[0]
    df_norm['XLF (Financials)'] = xlf['Close'].tail(norm_period) / xlf['Close'].tail(norm_period).iloc[0]
    
    st.line_chart(df_norm)
    st.caption("Kerberos often watches XBI (Biotech) as a 'canary in the coal mine' for risk appetite.")