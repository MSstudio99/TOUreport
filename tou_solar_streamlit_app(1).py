import io
import re
from datetime import datetime, date
from threading import RLock

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle

# Matplotlib can be unsafe with concurrent Streamlit sessions.
# A lock prevents figures from being mixed between users.
_PLOT_LOCK = RLock()

LABEL_FMT = "%a, %d-%b-%Y %H:%M"


# -----------------------------
# Helper functions
# -----------------------------
def extract_cabin_name(filename: str) -> str:
    """Extract a clean cabin name such as P3415 or PT1252 from uploaded filename."""
    stem = filename.rsplit(".", 1)[0]
    match = re.search(r"\b[A-Za-z]{1,5}\d{2,8}\b", stem)
    if match:
        return match.group(0).upper()
    return stem.split("_")[0].split("-")[0].upper()


def extract_numeric(value):
    """Extract first numeric value from values such as '171.841 kW'."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    return float(match.group(0)) if match else np.nan


def read_csv_flexible(uploaded_file) -> pd.DataFrame:
    """Read CSV with automatic separator detection, with fallback options."""
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)


def guess_column(columns, candidates):
    """Return best matching column name from candidate keywords."""
    normalized = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in normalized:
            return normalized[key]
    for col in columns:
        col_l = str(col).strip().lower()
        if any(c.strip().lower() in col_l for c in candidates):
            return col
    return columns[0] if len(columns) else None


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse datetime robustly. Keeps notebook format but falls back to pandas parser."""
    known_formats = [
        "%d-%b-%y %I:%M:%S %p",
        "%d-%b-%Y %I:%M:%S %p",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
    ]
    best = None
    best_valid = -1
    for fmt in known_formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        valid = int(parsed.notna().sum())
        if valid > best_valid:
            best = parsed
            best_valid = valid
    fallback = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if fallback.notna().sum() > best_valid:
        return fallback
    return best


def convert_to_kw(series: pd.Series, unit_mode: str) -> pd.Series:
    """Convert selected value column to kW."""
    numeric = series.apply(extract_numeric).astype(float)
    if unit_mode == "W → kW":
        return numeric / 1000.0
    return numeric


def process_daily_max(df: pd.DataFrame, datetime_col: str, value_col: str, unit_mode: str) -> pd.DataFrame:
    timestamps = parse_datetime_series(df[datetime_col])
    kw = convert_to_kw(df[value_col], unit_mode)

    data = (
        pd.DataFrame({"timestamp": timestamps, "kW": kw})
        .dropna(subset=["timestamp", "kW"])
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    if data.empty:
        return pd.DataFrame()

    grouped = data["kW"].groupby(data.index.date)
    idx_of_max = grouped.idxmax()
    val_of_max = grouped.max()

    result = pd.DataFrame(
        {
            "peak_timestamp": pd.to_datetime(idx_of_max.values),
            "daily_max_kW": val_of_max.values,
        },
        index=pd.to_datetime(idx_of_max.index),
    )
    result.index.name = "date"
    return result


def build_summary_table(filtered: pd.DataFrame, contract_kw: float) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "No.": range(1, len(filtered) + 1),
            "Date": filtered["peak_timestamp"].dt.strftime(LABEL_FMT),
            "Daily Max kW": filtered["daily_max_kW"].round(2),
            "Contract kW": round(contract_kw, 2),
        }
    )
    table["Excess kW"] = (table["Daily Max kW"] - table["Contract kW"]).clip(lower=0).round(2)
    table["Remaining kW"] = (table["Contract kW"] - table["Daily Max kW"]).round(2)
    table["Status"] = np.where(table["Excess kW"] > 0, "Over Contract", "OK")
    return table


def adaptive_tick(step_candidates, vmin, vmax):
    rng = max(1e-9, vmax - vmin)
    for step in step_candidates:
        if rng / step <= 12:
            return step
    return step_candidates[-1]


def make_chart(filtered: pd.DataFrame, cabin_name: str, contract_kw: float, start_date, end_date):
    labels = filtered["peak_timestamp"].dt.strftime(LABEL_FMT).tolist()
    yvals = filtered["daily_max_kW"].to_numpy(dtype=float)
    xpos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(xpos, yvals, marker="o", color="black", label="Daily Max (kW)")
    ax.axhline(contract_kw, linestyle="--", color="red", label=f"Contract {contract_kw:g} kW")

    for xi, yi in zip(xpos, yvals):
        ax.vlines(xi, min(yi, contract_kw), max(yi, contract_kw), linestyles="--", colors="red", alpha=0.35)

    max_idx = int(np.argmax(yvals))
    max_x = xpos[max_idx]
    max_y = yvals[max_idx]
    max_label = labels[max_idx]
    ax.annotate(
        f"Max: {max_y:.2f} kW\n{max_label}",
        xy=(max_x, max_y),
        xytext=(0, 25),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="gray", alpha=0.85),
    )

    ax.set_title(f"Daily Maximum Demand – {cabin_name}\n{start_date} to {end_date}")
    ax.set_ylabel("ACTIVE POWER (kW)")
    ax.set_xlabel("")

    ymin, ymax = min(np.min(yvals), contract_kw), max(np.max(yvals), contract_kw)
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(adaptive_tick([10, 20, 50, 100], ymin, ymax)))
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", transparent=True)
    buffer.seek(0)
    return buffer.getvalue()


def make_table_png(table: pd.DataFrame) -> bytes:
    highlight_color = "#fff2cc"
    display_df = table.copy()
    for col in ["Daily Max kW", "Contract kW", "Excess kW", "Remaining kW"]:
        display_df[col] = display_df[col].map(lambda v: f"{float(v):,.2f}")

    fig_h = max(3, len(display_df) * 0.35)
    fig, ax = plt.subplots(figsize=(11, fig_h + 0.6))
    ax.axis("off")

    col_widths = [0.25, 1.7, 0.9, 0.9, 0.9, 0.9, 0.8]
    norm_col_widths = [w / sum(col_widths) for w in col_widths]
    mpl_table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        colWidths=norm_col_widths,
        loc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(9)
    mpl_table.scale(1, 1.2)

    for (row, col), cell in mpl_table.get_celld().items():
        if row == 0:
            cell.set_linewidth(0.6)
            cell.set_edgecolor("black")
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")
        else:
            cell.set_linewidth(0.3)
            cell.set_edgecolor("#cccccc")

    if len(table) > 0:
        idx_high = int(table["Daily Max kW"].astype(float).values.argmax()) + 1
        for col in range(display_df.shape[1]):
            mpl_table[(idx_high, col)].set_facecolor(highlight_color)

    fig.subplots_adjust(bottom=0)
    fig.patches.append(
        Rectangle((0.1, 0.04), 0.02, 0.02, transform=fig.transFigure, facecolor=highlight_color, edgecolor="gray")
    )
    fig.text(0.13, 0.05, "Highest Daily Max kW", va="center", fontsize=10)

    png = fig_to_png_bytes(fig)
    plt.close(fig)
    return png


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Daily Max")
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="TOU Solar Daily Max Demand", layout="wide")

st.title("TOU Solar / Cabin Daily Maximum Demand Checker")
st.caption("Upload a CSV, calculate each day’s maximum kW, compare against contract kW, and export the result.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload one CSV file to begin.")
    st.stop()

try:
    raw_df = read_csv_flexible(uploaded_file)
except Exception as exc:
    st.error(f"Could not read this CSV file: {exc}")
    st.stop()

if raw_df.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

columns = list(raw_df.columns)
default_datetime_col = guess_column(columns, ["Date/Time", "DateTime", "Timestamp", "Date"])
default_value_col = guess_column(columns, ["Value", "Watt Total Avg", "Watt Total  Avg", "kW", "KW", "Active Power"])

with st.expander("Preview uploaded data", expanded=False):
    st.write(f"**Filename:** `{uploaded_file.name}`")
    st.dataframe(raw_df.head(20), use_container_width=True)

st.sidebar.header("Analysis settings")

cabin_name = st.sidebar.text_input("Cabin name", value=extract_cabin_name(uploaded_file.name))
contract_kw = st.sidebar.number_input("Contract kW", min_value=0.0, value=400.0, step=10.0)

datetime_col = st.sidebar.selectbox(
    "Date/time column",
    options=columns,
    index=columns.index(default_datetime_col) if default_datetime_col in columns else 0,
)
value_col = st.sidebar.selectbox(
    "Power value column",
    options=columns,
    index=columns.index(default_value_col) if default_value_col in columns else 0,
)

unit_mode = st.sidebar.radio(
    "Input unit",
    options=["Already kW", "W → kW"],
    index=1 if "watt" in str(value_col).lower() else 0,
    help="Use 'W → kW' when the source column is Watt Total Avg. Use 'Already kW' when values already contain kW.",
)

result = process_daily_max(raw_df, datetime_col, value_col, unit_mode)

if result.empty:
    st.error("No valid timestamp/kW rows found. Check the selected date/time and power columns.")
    st.stop()

min_date = result["peak_timestamp"].dt.date.min()
max_date = result["peak_timestamp"].dt.date.max()
selected_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date, end_date = min_date, max_date

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

mask = (
    (result["peak_timestamp"].dt.date >= start_date)
    & (result["peak_timestamp"].dt.date <= end_date)
)
filtered = result.loc[mask].copy()

if filtered.empty:
    st.warning("No data found inside the selected date range.")
    st.stop()

table = build_summary_table(filtered, contract_kw)

peak_row = filtered.loc[filtered["daily_max_kW"].idxmax()]
peak_kw = float(peak_row["daily_max_kW"])
peak_time = peak_row["peak_timestamp"]
exceed_days = int((table["Excess kW"] > 0).sum())
max_excess = float(table["Excess kW"].max())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Maximum kW", f"{peak_kw:,.2f}")
col2.metric("Peak time", peak_time.strftime(LABEL_FMT))
col3.metric("Days over contract", exceed_days)
col4.metric("Highest excess kW", f"{max_excess:,.2f}")

if exceed_days > 0:
    st.warning(f"{cabin_name} exceeded the contract on {exceed_days} day(s). Highest excess: {max_excess:.2f} kW.")
else:
    st.success(f"{cabin_name} stayed within the {contract_kw:g} kW contract for the selected period.")

st.subheader("Daily Maximum Demand Chart")
with _PLOT_LOCK:
    fig = make_chart(filtered, cabin_name, contract_kw, start_date, end_date)
    st.pyplot(fig, clear_figure=False)
    chart_png = fig_to_png_bytes(fig)
    plt.close(fig)

st.subheader("Daily Maximum Demand Table")
st.dataframe(
    table.style.apply(lambda row: ["background-color: #fff2cc" if row["Daily Max kW"] == table["Daily Max kW"].max() else "" for _ in row], axis=1),
    use_container_width=True,
)

table_png = make_table_png(table)
csv_bytes = dataframe_to_csv_bytes(table)
excel_bytes = dataframe_to_excel_bytes(table)

st.subheader("Export")
d1, d2, d3, d4 = st.columns(4)
d1.download_button(
    "Download chart PNG",
    data=chart_png,
    file_name=f"{cabin_name}_daily_max_chart.png",
    mime="image/png",
    use_container_width=True,
)
d2.download_button(
    "Download table PNG",
    data=table_png,
    file_name=f"{cabin_name}_daily_max_table.png",
    mime="image/png",
    use_container_width=True,
)
d3.download_button(
    "Download CSV summary",
    data=csv_bytes,
    file_name=f"{cabin_name}_daily_max_summary.csv",
    mime="text/csv",
    use_container_width=True,
)
d4.download_button(
    "Download Excel summary",
    data=excel_bytes,
    file_name=f"{cabin_name}_daily_max_summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

with st.expander("Data quality checks"):
    parsed_ts = parse_datetime_series(raw_df[datetime_col])
    parsed_kw = convert_to_kw(raw_df[value_col], unit_mode)
    st.write({
        "Total rows": int(len(raw_df)),
        "Valid timestamp rows": int(parsed_ts.notna().sum()),
        "Valid kW rows": int(parsed_kw.notna().sum()),
        "Rows used after cleaning": int(len(process_daily_max(raw_df, datetime_col, value_col, unit_mode))),
        "Source date min": str(min_date),
        "Source date max": str(max_date),
    })
