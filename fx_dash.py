import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FX Institutional Dashboard | EUR/USD",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to match the 'Dark/Institutional' aesthetic
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
    /* Stat Value Colors */
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
# 2. Mock Data Generation (Simulating the visual patterns)
# -----------------------------------------------------------------------------
@st.cache_data
def generate_mock_data():
    # Date range: 2 years back to present + small forecast
    dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
    
    # 1. Price (EUR/USD) - Random walk with trend
    np.random.seed(42)
    price_noise = np.random.normal(0, 0.005, size=len(dates))
    price = 1.05 + np.cumsum(price_noise)
    price = price + np.linspace(0, 0.1, len(dates)) # Slight drift
    
    # 2. Macro Spread (Smoothed) - Green/Red waves
    macro_raw = np.cumsum(np.random.normal(0, 0.05, size=len(dates)))
    macro_smoothed = pd.Series(macro_raw).rolling(window=20).mean().fillna(0)
    
    # 3. Correlation (Oscillator -1 to 1)
    correlation = np.sin(np.linspace(0, 20, len(dates))) * 0.8 + np.random.normal(0, 0.1, size=len(dates))
    
    # 4. COT (Positioning) - Diverging lines
    cot_net = 200000 + np.cumsum(np.random.normal(0, 1000, size=len(dates)))
    # Clip to create the "limit" effect seen in image
    cot_net = np.clip(cot_net, 150000, 240000) 
    cot_band_top = np.full(len(dates), 268000)
    cot_band_bot = np.full(len(dates), 130000)
    
    # 5. Volatility (Spikey)
    vol_base = 5 + np.random.gamma(2, 1, size=len(dates))
    # Add a "compression" period at the end
    vol_base[-50:] = 3.59 + np.random.normal(0, 0.2, size=50)
    
    df = pd.DataFrame({
        'Date': dates,
        'Price': price,
        'Macro': macro_smoothed,
        'Correlation': correlation,
        'COT_Net': cot_net,
        'COT_Top': cot_band_top,
        'COT_Bot': cot_band_bot,
        'Volatility': vol_base
    })
    return df

df = generate_mock_data()

# -----------------------------------------------------------------------------
# 3. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("FX Hero Control")
    st.markdown("---")
    asset = st.selectbox("Asset Class", ["EUR/USD", "GBP/USD", "USD/JPY"], index=0)
    timeframe = st.selectbox("Timeframe", ["Daily", "Weekly", "4H"], index=0)
    
    st.markdown("### Model Settings")
    st.slider("Macro Lookback", 10, 200, 50)
    st.slider("Vol Smoothing", 5, 50, 14)
    
    st.info("Based on 'From Zero to FX Hero' methodology. Data is simulated for demonstration.")

# -----------------------------------------------------------------------------
# 4. Header & Executive Summary (The Verdict)
# -----------------------------------------------------------------------------
st.title(f"Institutional Dashboard: {asset}")
st.markdown("### Executive Summary (The Verdict)")

col_verdict_1, col_verdict_2 = st.columns([2, 1])

with col_verdict_1:
    st.markdown("""
    **Current Regime:** :orange[Wait & See / Breakout Watch]
    
    * **The Trap:** Do not fade (short) the move just yet. Even though the crowd is long (**233k**), they haven't hit the statistical "Max Long" cap (**268k**). There is no "overbought" signal confirmed by the bands yet.
    * **The Risk:** With volatility at **~3.6%**, the market is coiled like a spring. A breakout from the current range is imminent.
    * **The Play:** Since the Macro Spread is flat and Correlation is weak, **do not trade the carry**. Wait for volatility to expand.
    """)

with col_verdict_2:
    st.markdown("""
    **Scenario Watch:**
    * 🟢 **Bullish Case:** If price breaks higher, room for COT to expand to 268k.
    * 🔴 **Bearish Case:** If Macro Spread turns red, crowded longs will liquidate.
    """)

st.divider()

# -----------------------------------------------------------------------------
# 5. KPI Metrics Row (Replicating Image 2 Table)
# -----------------------------------------------------------------------------
# Helper function for metric cards
def metric_card(title, value, status, implication, color_code="#ffffff"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="stat-label">{title}</div>
        <div class="stat-value" style="color: {color_code}">{value}</div>
        <div class="stat-sub" style="font-weight:bold; margin-top:5px;">{status}</div>
        <div style="font-size: 12px; color: #aaa; margin-top: 8px; line-height: 1.2;">{implication}</div>
    </div>
    """, unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    metric_card(
        "MACRO ENGINE", 
        "1.55% (Flat)", 
        "NEUTRAL / CARRY", 
        "Rate differential favors USD, but momentum stalled. 'Magnet' is off.",
        "#AAAAAA" # Grey/Flat color
    )

with kpi2:
    metric_card(
        "POSITIONING", 
        "233k Net Long", 
        "CROWDED (Room to Run)", 
        "Speculators heavily long, but not at extreme (268k band).",
        "#FFA500" # Orange for crowded
    )

with kpi3:
    metric_card(
        "VOLATILITY", 
        "3.59%", 
        "DEEP COMPRESSION", 
        "Market is asleep. Expect explosive breakout soon.",
        "#FF4B4B" # Red for danger/alert
    )

with kpi4:
    metric_card(
        "ALIGNMENT", 
        "0.30", 
        "DECOUPLED", 
        "Price ignoring rates. Driven by technical flows/option pinning.",
        "#888888"
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. Technical Charts (Replicating Image 1 Stack)
# -----------------------------------------------------------------------------
st.subheader("Technical & Macro Structure")

# Create subplot with shared X-axis
fig = make_subplots(
    rows=5, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.3, 0.15, 0.15, 0.20, 0.20],
    subplot_titles=(
        "Price Action (1D)", 
        "FX Institutional Macro Spread (Smoothed)", 
        "Correlation: Price vs Yield Spread", 
        "COT Total Positioning", 
        "FX Realized Volatility (Annualized)"
    )
)

# 1. Price Chart (Candlestick simulation using Line for cleaner look in demo)
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Price'],
    line=dict(color='white', width=1),
    name="EUR/USD"
), row=1, col=1)

# 2. Macro Spread (Color coded line based on value/slope is tricky in simple plotly, using Gradient simulation)
# We will use a scatter with marker colors to simulate the green/red/grey line
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Macro'],
    line=dict(color='#A0A0A0', width=2), # Base grey
    name="Macro Spread",
    fill='tozeroy', # Fill to show magnitude
    fillcolor='rgba(100, 100, 100, 0.1)'
), row=2, col=1)

# 3. Correlation
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Correlation'],
    line=dict(color='#00FF7F', width=1.5),
    name="Correlation"
), row=3, col=1)
# Zero line for correlation
fig.add_shape(type="line",
    x0=df['Date'].iloc[0], y0=0, x1=df['Date'].iloc[-1], y1=0,
    line=dict(color="white", width=0.5, dash="dash"),
    row=3, col=1
)

# 4. COT Positioning (Bands + Net)
# Top Band
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['COT_Top'],
    line=dict(color='rgba(255, 100, 100, 0.5)', width=1, dash='dot'),
    name="Max Long Cap"
), row=4, col=1)
# Net Position
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['COT_Net'],
    line=dict(color='#3399FF', width=2),
    name="Net Longs"
), row=4, col=1)
# Bottom Band
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['COT_Bot'],
    line=dict(color='rgba(100, 100, 255, 0.5)', width=1, dash='dot'),
    name="Min Cap"
), row=4, col=1)

# 5. Volatility
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Volatility'],
    line=dict(color='#E066FF', width=1.5),
    name="Realized Vol"
), row=5, col=1)
# Warning line at current low vol
fig.add_shape(type="line",
    x0=df['Date'].iloc[0], y0=3.6, x1=df['Date'].iloc[-1], y1=3.6,
    line=dict(color="red", width=1, dash="dot"),
    row=5, col=1
)

# Layout Updates
fig.update_layout(
    height=900,
    template="plotly_dark",
    margin=dict(l=20, r=20, t=40, b=20),
    showlegend=False,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

# Update Axis Formatting
fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor='#333')
fig.update_xaxes(showgrid=False)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. Additional Analysis Footer
# -----------------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px; margin-top: 30px;">
    <i>Dashboard based on "From Zero to FX Hero" methodology.</i><br>
    <i>Data provided for simulation purposes only. Not financial advice.</i>
</div>
""", unsafe_allow_html=True)