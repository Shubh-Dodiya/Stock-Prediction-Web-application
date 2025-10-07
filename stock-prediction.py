# stock-prediction_fixed.py
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import io, math

st.set_page_config(
    page_title="Stock Prediction",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------
# Helpers
# ----------------------
def flatten_multiindex_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # join levels sensibly (skip empty strings)
        new_cols = []
        for col in df.columns:
            # col might be tuple like ('AAPL', 'Close') or ('', 'Close')
            parts = [str(p) for p in col if (p is not None and str(p) != "")]
            new_cols.append("_".join(parts) if parts else "")
        df = df.copy()
        df.columns = new_cols
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_and_rename_date_close(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = flatten_multiindex_cols(df)

    # 1) Try find obvious date column names
    date_candidates = [c for c in df.columns if any(k in c.lower() for k in ("date", "time", "datetime", "timestamp", "index"))]
    date_col = date_candidates[0] if date_candidates else None

    # 2) If not found, look for datetime dtype
    if date_col is None:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_col = c
                break

    # 3) If still not found, attempt parsing each column and pick best
    if date_col is None:
        best = None
        best_nonnull = -1
        for c in df.columns:
            # skip obviously numeric/price columns
            if pd.api.types.is_numeric_dtype(df[c]):
                continue
            parsed = pd.to_datetime(df[c], errors="coerce")
            nonnull = int(parsed.notna().sum())
            if nonnull > best_nonnull:
                best_nonnull = nonnull
                best = c
        # require at least some parsed dates
        if best_nonnull > 0:
            date_col = best

    # Apply date parsing if we found a candidate
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df.rename(columns={date_col: "Date"}, inplace=True)

    # ----------------
    # Find Close column
    close_candidates = [c for c in df.columns if "close" in c.lower() or "adj close" in c.lower() or "adjclose" in c.lower()]
    close_col = close_candidates[0] if close_candidates else None

    # If not found, pick the last numeric column (often Close)
    if close_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        # exclude volume if present
        numeric_cols = [c for c in numeric_cols if "vol" not in c.lower()]
        if numeric_cols:
            close_col = numeric_cols[-1]  # fallback: last numeric column

    if close_col:
        df.rename(columns={close_col: "Close"}, inplace=True)

    return df

def safe_to_numeric(series):
    try:
        return pd.to_numeric(series, errors="coerce").astype(float)
    except Exception:
        return pd.Series(dtype=float)

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"mae": None, "rmse": None}
    mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
    rmse = math.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
    return {"mae": mae, "rmse": rmse}

# ----------------------
# Data loader with normalization
# ----------------------
@st.cache_data(show_spinner=False)
def load_yf_data_normalized(symbol, start, end, interval="1d"):
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    if df is None:
        return pd.DataFrame()
    if df.empty and hasattr(df, "empty"):
        return pd.DataFrame()
    # If index is datetime-like, reset it so we get a column
    df = df.reset_index()
    # Normalize column names & flatten multiindex
    df = find_and_rename_date_close(df)
    return df

# ----------------------
# Sidebar Controls
# ----------------------
with st.sidebar:
    st.header("Classic App Settings")
    symbol = st.selectbox("Ticker Symbol (e.g. AAPL)", ["AAPL"], index=0)
    start_date = st.date_input("Start Date", value=date(2015, 1, 1))
    end_date = st.date_input("End Date", value=date.today())
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    forecast_days = st.slider("Forecast Days (horizon)", 30, 1095, 365)
    yearly_seasonality = st.checkbox("Enable yearly seasonality", value=True)
    dark_mode = st.checkbox("Dark Mode (classic look)", value=True)

# ----------------------
# Load data
# ----------------------
data = load_yf_data_normalized(symbol.upper(), start_date, end_date, interval=interval)

# If still empty or missing columns - show helpful diagnostic
if data.empty:
    st.error("No data returned from yfinance for this symbol/period. Try another symbol or date range.")
    st.stop()

# Check for Date column
if "Date" not in data.columns:
    # helpful debug output: show columns and first few rows to inspect
    st.error("Missing 'Date' column after normalization.")
    st.write("Columns found:", list(data.columns))
    st.write("Preview of first rows:")
    st.dataframe(data.head(10))
    st.stop()

# Check for Close column
if "Close" not in data.columns:
    st.error("Missing 'Close' column after normalization.")
    st.write("Columns found:", list(data.columns))
    st.write("Preview of first rows:")
    st.dataframe(data.head(10))
    st.stop()

# Clean and sort
data["Close"] = safe_to_numeric(data["Close"])
data = data.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

if data.empty:
    st.error("No valid rows after cleaning Date/Close.")
    st.stop()

# ----------------------
# UI and plotting
# ----------------------
st.title("📈 Stock Prediction")
st.subheader(f"Data for {symbol.upper()} ({len(data)} rows)")

st.markdown("**Preview (last 5 rows):**")
st.dataframe(data.tail())

fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", mode="lines"))
fig.update_layout(
    title=f"{symbol.upper()} Close Price",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark" if dark_mode else "plotly_white",
    xaxis_rangeslider_visible=True,
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Prophet forecasting
# ----------------------
df_prophet = pd.DataFrame({"ds": pd.to_datetime(data["Date"]), "y": data["Close"].astype(float)})
df_prophet = df_prophet.dropna().reset_index(drop=True)

st.subheader("🔮 Forecast")
# train on full history for demo
try:
    model = Prophet(yearly_seasonality=yearly_seasonality)
    model.fit(df_prophet)
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

st.markdown("**Forecast preview:**")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

# interactive forecast plot
fc_plot = go.Figure()
fc_plot.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], name="Historical", mode="lines"))
fc_plot.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", mode="lines"))
fc_plot.add_trace(
    go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(100,150,200,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Confidence Interval",
    )
)
fc_plot.update_layout(template="plotly_dark" if dark_mode else "plotly_white", title=f"{symbol.upper()} Forecast")
st.plotly_chart(fc_plot, use_container_width=True)

# allow download
csv_buf = io.StringIO()
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(csv_buf, index=False)
st.download_button("Download forecast CSV", data=csv_buf.getvalue().encode(), file_name=f"{symbol}_forecast.csv")

st.success("Done. If you still see 'Missing Date' message, inspect the table shown above — it helps find which column to treat as date.")
