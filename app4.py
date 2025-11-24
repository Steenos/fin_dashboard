import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas_datareader.data as web
import datetime
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="F&G Pro Suite", layout="wide", page_icon="🐺")

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
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
    }
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
        "RSP": "RSP",   # Equal Weight SP500
        "SPY": "SPY",   # Proxy for Volume
        # Sectors
        "XLK": "XLK", "XLF": "XLF", "XLV": "XLV", "XLE": "XLE", 
        "XLY": "XLY", "XLP": "XLP", "XLI": "XLI", "XLU": "XLU"
    }
    
    data = {}
    try:
        # Download all at once
        raw_df = yf.download(list(tickers.values()), period=period, group_by='ticker', progress=False)
        
        for key, ticker in tickers.items():
            try:
                df = raw_df[ticker].copy()
                if df.empty: continue
                
                # Basic Cleaning
                df = df.dropna()
                
                # Technicals: Bollinger Bands & %B
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['StdDev'] = df['Close'].rolling(window=20).std()
                df['Upper_BB'] = df['SMA_20'] + (2 * df['StdDev'])
                df['Lower_BB'] = df['SMA_20'] - (2 * df['StdDev'])
                # Avoid division by zero
                denom = df['Upper_BB'] - df['Lower_BB']
                df['Pct_B'] = np.where(denom == 0, 0, (df['Close'] - df['Lower_BB']) / denom)
                
                # Returns
                df['Returns'] = df['Close'].pct_change()
                
                data[key] = df
            except KeyError:
                continue
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
            
    return data

@st.cache_data(ttl=3600)
def fetch_macro_data():
    """Fetches Official Yield & Liquidity Data from FRED."""
    start = datetime.datetime.now() - datetime.timedelta(days=365*2)
    end = datetime.datetime.now()
    
    # FRED Codes:
    # DGS10: 10-Year Treasury Yield
    # T10Y2Y: 10Y-2Y Spread
    # WALCL: Fed Total Assets (Balance Sheet)
    # WTREGEN: Treasury General Account (TGA)
    # RRPONTSYD: Reverse Repo (RRP)
    
    series = ['DGS10', 'T10Y2Y', 'WALCL', 'WTREGEN', 'RRPONTSYD']
    try:
        df = web.DataReader(series, 'fred', start, end)
        df = df.ffill()
        
        # Calculate Net Liquidity
        # Formula: Fed Assets - TGA - RRP
        # Note: Units might differ (Mil vs Bil). Usually FRED sends Millions.
        # WALCL (Mil), WTREGEN (Bil), RRP (Bil). Need to normalize to Billions.
        
        # Checking typical values:
        # WALCL ~ 7,000,000 (Mil) -> 7,000 Bil
        # WTREGEN ~ 700 (Bil)
        # RRP ~ 500 (Bil)
        
        df['Net_Liquidity'] = (df['WALCL'] / 1000) - df['WTREGEN'] - df['RRPONTSYD']
        
        return df
    except Exception as e:
        st.error(f"Error fetching FRED data: {e}")
        return pd.DataFrame()

# --- LOGIC MODULES ---

def calculate_no_demand(df):
    """'No Demand' Logic"""
    df = df.copy()
    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Price_Up'] = df['Close'] > df['Open']
    df['Price_Down'] = df['Close'] < df['Open']
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    
    df['No_Demand'] = (df['Price_Up']) & (df['Volume'] < df['Vol_SMA20'] * 0.65) & (df['Close'] >= df['Prev_High'])
    df['High_Vol_Dump'] = (df['Price_Down']) & (df['Volume'] > df['Vol_SMA20'] * 1.5) & (df['Close'] <= df['Prev_Low'])
    return df

def run_market_scan(tickers):
    """Scans a list of tickers for No Demand / Dump signals"""
    results = []
    # Force download 6mo data
    try:
        data = yf.download(tickers, period="6mo", group_by='ticker', progress=False)
        for ticker in tickers:
            try:
                df = data[ticker].copy()
                if df.empty: continue
                df = df.dropna()
                if len(df) < 20: continue # Need enough data for SMA
                
                df = calculate_no_demand(df)
                last = df.iloc[-1]
                
                status = "Neutral"
                if last['No_Demand']:
                    status = "🔴 No Demand (Trap)"
                elif last['High_Vol_Dump']:
                    status = "🟣 High Vol Dump"
                
                if status != "Neutral":
                    results.append({
                        "Ticker": ticker,
                        "Signal": status,
                        "Price": f"${last['Close']:.2f}",
                        "Vol": f"{(last['Volume']/last['Vol_SMA20']):.1f}x Avg"
                    })
            except:
                continue
    except Exception as e:
        return pd.DataFrame()
        
    return pd.DataFrame(results)

def run_backtest(df_vix, df_spx, buy_thresh, sell_thresh, initial_capital=10000):
    """Vectorized Backtest of FGMR Strategy"""
    # Align Data
    common = df_vix.index.intersection(df_spx.index)
    vix = df_vix.loc[common].copy()
    spx = df_spx.loc[common].copy()
    
    # Signals
    # Buy (Go Long SPX) when VIX > Buy Thresh (Panic)
    # Sell (Go Cash) when VIX < Sell Thresh (Greed)
    
    vix['Signal'] = 0
    vix.loc[vix['Pct_B'] > buy_thresh, 'Signal'] = 1 # Enter Long
    vix.loc[vix['Pct_B'] < sell_thresh, 'Signal'] = -1 # Exit
    
    # Fill signal forward (Holding period)
    vix['Position'] = vix['Signal'].replace(0, np.nan).ffill().fillna(0)
    # If Position is -1 (Exit), map to 0 (Cash) for calculation
    vix['Position'] = vix['Position'].replace(-1, 0)
    
    # Calculate Returns
    spx['Strat_Ret'] = spx['Returns'] * vix['Position'].shift(1) # Shift 1 to trade next open
    spx['BuyHold_Ret'] = spx['Returns']
    
    # Equity Curve
    spx['Strat_Equity'] = initial_capital * (1 + spx['Strat_Ret']).cumprod()
    spx['BuyHold_Equity'] = initial_capital * (1 + spx['BuyHold_Ret']).cumprod()
    
    return spx

# --- PLOTTING ---

def plot_net_liquidity(df_liq, df_spx):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Align dates
    common = df_liq.index.intersection(df_spx.index)
    liq = df_liq.loc[common]
    spx = df_spx.loc[common]
    
    fig.add_trace(go.Scatter(x=liq.index, y=liq['Net_Liquidity'], name="Net Liquidity (Fed)", line=dict(color='#00E5FF', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=spx.index, y=spx['Close'], name="S&P 500", line=dict(color='white', width=1, dash='dot')), secondary_y=True)
    
    fig.update_layout(title="Smart Money: Net Liquidity vs SPX", template="plotly_dark", height=500, hovermode="x unified")
    fig.update_yaxes(title_text="Liquidity ($B)", secondary_y=False)
    fig.update_yaxes(title_text="S&P 500", secondary_y=True)
    return fig

def plot_sector_rotation(market_data):
    # Calculate Relative Strength vs SPY over last 20 days
    sectors = ["XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLU"]
    spy = market_data['SPY']['Close']
    
    performance = {}
    
    for sec in sectors:
        if sec in market_data:
            sec_close = market_data[sec]['Close']
            # Align
            common = sec_close.index.intersection(spy.index)
            rel_perf = (sec_close.loc[common] / sec_close.loc[common].iloc[0]) / \
                       (spy.loc[common] / spy.loc[common].iloc[0])
            performance[sec] = (rel_perf.iloc[-1] - 1) * 100
            
    df_perf = pd.DataFrame(list(performance.items()), columns=['Sector', 'Rel_Perf'])
    df_perf = df_perf.sort_values('Rel_Perf', ascending=False)
    
    fig = px.bar(df_perf, x='Rel_Perf', y='Sector', orientation='h', 
                 color='Rel_Perf', color_continuous_scale='RdYlGn',
                 title="Sector Relative Strength vs SPY (Lookback Period)")
    fig.update_layout(template="plotly_dark", height=400)
    return fig

# --- MAIN APP LAYOUT ---

st.sidebar.title("Pro")
st.sidebar.markdown("---")
lookback = st.sidebar.select_slider("Lookback", ["6mo", "1y", "2y", "5y"], value="1y")

# Settings
with st.sidebar.expander("Strategy Settings"):
    vix_buy = st.number_input("VIX Panic Buy (> %B)", 1.0, 2.0, 1.0)
    vix_sell = st.number_input("VIX Greed Sell (< %B)", -1.0, 0.5, 0.0)

with st.sidebar.expander("🔔 Alert Configuration"):
    webhook_url = st.text_input("Discord Webhook URL")
    if st.button("Test Alert"):
        if webhook_url:
            try:
                payload = {"content": "🐺 **Pro**: This is a test alert from your dashboard."}
                requests.post(webhook_url, json=payload)
                st.success("Sent!")
            except:
                st.error("Failed to send.")
        else:
            st.warning("Enter a URL first.")

# FETCH DATA
with st.spinner("Initializing Pro Suite..."):
    market_data = fetch_market_data(lookback)
    macro_data = fetch_macro_data()

if not market_data:
    st.stop()

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚨 FGMR Signal", 
    "🔍 Pro Scanner", 
    "🏦 Net Liquidity", 
    "⚖️ Sectors", 
    "📈 Backtest", 
    "📉 Macro"
])

# --- TAB 1: FGMR ---
with tab1:
    vix = market_data['VIX']
    curr_b = vix['Pct_B'].iloc[-1]
    
    # Term Structure
    if 'VIX3M' in market_data:
        common = vix.index.intersection(market_data['VIX3M'].index)
        ratio = vix.loc[common]['Close'] / market_data['VIX3M'].loc[common]['Close']
        curr_ratio = ratio.iloc[-1]
    else:
        curr_ratio = 0
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if curr_b > vix_buy:
            st.markdown(f"<div class='metric-box'><h3>ACTION: BUY DIP</h3><p>VIX %B {curr_b:.2f} (Panic)</p></div>", unsafe_allow_html=True)
        elif curr_b < vix_sell:
            st.markdown(f"<div class='metric-box-bear'><h3>ACTION: SELL RIP</h3><p>VIX %B {curr_b:.2f} (Greed)</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='metric-box-warn'><h3>ACTION: HOLD</h3><p>VIX %B {curr_b:.2f} (Neutral)</p></div>", unsafe_allow_html=True)
            
    with c2:
        st.metric("VIX Term Structure", f"{curr_ratio:.2f}", delta="Panic (>1.0)" if curr_ratio > 1 else "Normal", delta_color="inverse")
    with c3:
        st.metric("SPX Price", f"{market_data['SPX']['Close'].iloc[-1]:.2f}")

    # Chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=vix.index, y=vix['Close'], name="VIX", line=dict(color='white')), row=1, col=1)
    fig.add_trace(go.Scatter(x=vix.index, y=vix['Upper_BB'], name="Upper", line=dict(dash='dot', color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(x=vix.index, y=vix['Lower_BB'], name="Lower", line=dict(dash='dot', color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(x=vix.index, y=vix['Pct_B'], name="%B", line=dict(color='#00E5FF')), row=2, col=1)
    fig.add_hline(y=vix_buy, row=2, col=1, line_color="green")
    fig.add_hline(y=vix_sell, row=2, col=1, line_color="red")
    fig.update_layout(template="plotly_dark", height=500, title="VIX Panic/Greed Cycle")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: SCANNER ---
with tab2:
    st.subheader("Market-Wide Anomaly Scanner")
    st.caption("Scans the 'Magnificent 7', Major ETFs, and Semis for Volume Traps.")
    
    scan_list = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", # Mag 7
                 "AMD", "INTC", "QCOM", "MU", # Semis
                 "QQQ", "IWM", "DIA", "TLT", "GLD", "SLV"] # ETFs
    
    if st.button("RUN LIVE SCAN"):
        with st.spinner("Analyzing volume profiles..."):
            results = run_market_scan(scan_list)
            if not results.empty:
                st.dataframe(results, use_container_width=True)
            else:
                st.success("No anomalies detected in watchlist.")

# --- TAB 3: NET LIQUIDITY ---
with tab3:
    st.subheader("Federal Reserve Net Liquidity")
    st.markdown("Tracks the **real** money supply: `Fed Balance Sheet - TGA - Reverse Repo`. When this Line (Blue) drops, Stocks (White) usually follow.")
    
    if not macro_data.empty:
        st.plotly_chart(plot_net_liquidity(macro_data, market_data['SPX']), use_container_width=True)
        
        curr_liq = macro_data['Net_Liquidity'].iloc[-1]
        prev_liq = macro_data['Net_Liquidity'].iloc[-20] # Month ago
        delta = curr_liq - prev_liq
        st.metric("Current Net Liquidity", f"${curr_liq:.2f}B", f"{delta:.2f}B (20d Change)")
    else:
        st.warning("Macro data unavailable.")

# --- TAB 4: SECTORS ---
with tab4:
    st.subheader("Sector Rotation (Risk On/Off)")
    st.plotly_chart(plot_sector_rotation(market_data), use_container_width=True)
    st.caption("Green bars = Outperforming SPY. Red bars = Underperforming.")

# --- TAB 5: BACKTEST ---
with tab5:
    st.subheader("Strategy Backtest (FGMR)")
    st.markdown(f"Simulating: **Buy SPX** when VIX > {vix_buy} (Panic), **Go Cash** when VIX < {vix_sell} (Greed).")
    
    if st.button("Run Simulation"):
        bt_results = run_backtest(market_data['VIX'], market_data['SPX'], vix_buy, vix_sell)
        
        # Metrics
        total_ret = (bt_results['Strat_Equity'].iloc[-1] / 10000 - 1) * 100
        bh_ret = (bt_results['BuyHold_Equity'].iloc[-1] / 10000 - 1) * 100
        
        b1, b2 = st.columns(2)
        b1.metric("Strategy Return", f"{total_ret:.2f}%")
        b2.metric("Buy & Hold Return", f"{bh_ret:.2f}%")
        
        # Plot
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_results.index, y=bt_results['Strat_Equity'], name="FGMR Strategy", line=dict(color='#00E5FF')))
        fig_bt.add_trace(go.Scatter(x=bt_results.index, y=bt_results['BuyHold_Equity'], name="Buy & Hold", line=dict(color='gray', dash='dot')))
        fig_bt.update_layout(title="Equity Curve ($10k Start)", template="plotly_dark", height=500)
        st.plotly_chart(fig_bt, use_container_width=True)

# --- TAB 6: MACRO ---
with tab6:
    if not macro_data.empty:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("10Y Yield", f"{macro_data['DGS10'].iloc[-1]:.2f}%")
        with col_m2:
            st.metric("Yield Curve (10Y-2Y)", f"{macro_data['T10Y2Y'].iloc[-1]:.2f}%")
        st.line_chart(macro_data[['DGS10', 'T10Y2Y']])