# stock-prediction-multi.py
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import io, math

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="📈 Stock Prediction Dashboard",
    page_icon="💹",
    layout="wide",
)

# -------------------------------
# Helper Functions
# -------------------------------
def flatten_multiindex_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flattens multi-index columns returned by yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in tup if c]).strip() for tup in df.columns]
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_and_rename_date_close(df: pd.DataFrame) -> pd.DataFrame:
    """Finds and renames Date and Close columns automatically."""
    df = flatten_multiindex_cols(df)
    date_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
    if not date_col:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_col = c
                break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df.rename(columns={date_col: "Date"}, inplace=True)

    close_col = next((c for c in df.columns if "close" in c.lower()), None)
    if not close_col:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and "vol" not in c.lower()]
        if numeric_cols:
            close_col = numeric_cols[-1]
    if close_col:
        df.rename(columns={close_col: "Close"}, inplace=True)
    return df

def safe_to_numeric(series):
    try:
        return pd.to_numeric(series, errors="coerce").astype(float)
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def load_yf_data(symbol, start, end, interval="1d"):
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df = find_and_rename_date_close(df)
    return df

# -------------------------------
# Sidebar Controls
# -------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    stock_choice = st.selectbox(
        "Select a Stock",
        ["GOOGL", "AAPL", "AMZN", "TSLA", "MSFT", "Other (Manual Input)"],
        index=0
    )

    if stock_choice == "Other (Manual Input)":
        symbol = st.text_input("Enter Custom Symbol (e.g., INFY.NS)", value="")
    else:
        symbol = stock_choice

    start_date = st.date_input("📅 Start Date", value=date(2015, 1, 1))
    end_date = st.date_input("📅 End Date", value=date.today())
    interval = st.selectbox("📊 Interval", ["1d", "1wk", "1mo"], index=0)
    forecast_days = st.slider("🔮 Forecast Days", 30, 1095, 365)
    yearly_seasonality = st.checkbox("Enable Yearly Seasonality", True)
    dark_mode = st.checkbox("Dark Mode", True)

# -------------------------------
# Data Loading
# -------------------------------
if not symbol:
    st.warning("Please select or enter a stock symbol to begin.")
    st.stop()

data = load_yf_data(symbol.upper(), start_date, end_date, interval)
if data.empty:
    st.error("⚠️ No data found. Try another symbol or time range.")
    st.stop()

if "Date" not in data.columns or "Close" not in data.columns:
    st.error("Data missing required columns (Date/Close).")
    st.dataframe(data.head())
    st.stop()

data["Close"] = safe_to_numeric(data["Close"])
data = data.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

# -------------------------------
# Tabs: Historical | Forecast
# -------------------------------
st.title("📈 Stock Prediction Dashboard")
tab1, tab2 = st.tabs(["📊 Historical Data", "🔮 Forecast Results"])

# -------------------------------
# Historical Tab
# -------------------------------
with tab1:
    st.subheader(f"Data for {symbol.upper()} ({len(data)} rows)")
    st.markdown("**Preview (last 5 rows):**")
    st.dataframe(data.tail().reset_index(drop=True), use_container_width=True)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=data["Date"], y=data["Close"], 
        name="Close", mode="lines", line=dict(width=2.2)
    ))
    fig_hist.update_layout(
        title=f"{symbol.upper()} Close Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark" if dark_mode else "plotly_white",
        xaxis_rangeslider_visible=True,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------------
# Forecast Tab
# -------------------------------
with tab2:
    df_prophet = pd.DataFrame({"ds": pd.to_datetime(data["Date"]), "y": data["Close"].astype(float)}).dropna()

    with st.spinner("⏳ Training Prophet model..."):
        try:
            model = Prophet(yearly_seasonality=yearly_seasonality)
            model.fit(df_prophet)
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    st.markdown("**Forecast Preview (last 5 rows):**")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail().reset_index(drop=True),
                 use_container_width=True)

    fig_fore = go.Figure()
    fig_fore.add_trace(go.Scatter(
        x=df_prophet["ds"], y=df_prophet["y"], name="Historical", mode="lines"
    ))
    fig_fore.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"], name="Forecast", mode="lines"
    ))
    fig_fore.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(100,150,200,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Confidence Interval",
    ))
    fig_fore.update_layout(
        title=f"{symbol.upper()} — Forecast ({forecast_days} Days)",
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_fore, use_container_width=True)

    # Download section
    st.markdown("### 💾 Download Forecast Data")
    csv_buf = io.StringIO()
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(csv_buf, index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv_buf.getvalue().encode(),
        file_name=f"{symbol}_forecast.csv"
    )

st.success("✅ Forecast completed successfully. Enjoy your data")
