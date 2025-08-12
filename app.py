import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from fpdf import FPDF
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

# ------------------------- Config & Theme -------------------------

st.set_page_config(page_title="Refined Data Profiling & Cleaning", layout="wide")

# Colors (consistent theme)
HEADING_COLOR = "#0A81D1"
UI_BLUE = "#F8D7DA"  # soft pink
HIGHLIGHT_TEXT = "#721C24"  # dark maroon
CARD_BG = "#F4F6F8"

# -------------------- Matplotlib PDF Helpers ----------------------
def add_matplotlib_chart_to_pdf(data, chart_type, pdf, title):
    """
    data: pd.Series or pd.DataFrame for scatter
    chart_type: 'hist','pie','bar','scatter'
    pdf: FPDF object
    title: string
    """
    try:
        fig, ax = plt.subplots(figsize=(6,4))
        if chart_type == "hist":
            ax.hist(data.dropna(), bins=20)
            ax.set_xlabel(data.name if hasattr(data, 'name') else 'Value')
            ax.set_ylabel("Frequency")
        elif chart_type == "pie":
            counts = data.value_counts()
            ax.pie(counts.values, labels=counts.index.astype(str), autopct='%1.1f%%')
        elif chart_type == "bar":
            counts = data.value_counts().head(10)
            ax.bar(counts.index.astype(str), counts.values)
            plt.xticks(rotation=45, ha='right')
        elif chart_type == "scatter":
            if isinstance(data, pd.DataFrame) and data.shape[1] >= 2:
                ax.scatter(data.iloc[:,0], data.iloc[:,1], alpha=0.7)
                ax.set_xlabel(data.columns[0])
                ax.set_ylabel(data.columns[1])
        ax.set_title(title)
        plt.tight_layout()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(tmp.name, bbox_inches='tight')
        plt.close(fig)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, title, ln=True)
        pdf.image(tmp.name, w=180)
        pdf.ln(5)
    except Exception as e:
        # Write a note in PDF
        pdf.set_font("Arial", 'I', 10)
        pdf.ln(5)
        pdf.cell(0, 10, f"[Could not render '{title}' - {e}]")
        pdf.ln(5)


def add_matplotlib_correlation_heatmap(df_numeric, pdf, title):
    try:
        fig, ax = plt.subplots(figsize=(6,5))
        corr = df_numeric.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(tmp.name, bbox_inches='tight')
        plt.close(fig)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, title, ln=True)
        pdf.image(tmp.name, w=180)
        pdf.ln(5)
    except Exception as e:
        pdf.set_font("Arial", 'I', 10)
        pdf.ln(5)
        pdf.cell(0, 10, f"[Could not render '{title}' - {e}]")
        pdf.ln(5)

# ------------------------- Helpers -------------------------------
def detect_delimiter(file_bytes):
    try:
        sample = file_bytes.read(5000).decode('utf-8')
        dlms = [',', ';', '\t', '|']
        counts = [sample.count(d) for d in dlms]
        return dlms[np.argmax(counts)]
    except Exception:
        return ','

# Robust cast function for custom fills
def try_cast_fill(value_str, dtype):
    """Attempt to cast custom input string to given dtype. Return (success, value_or_msg)."""
    if value_str is None:
        return False, None
    v = value_str
    try:
        if pd.api.types.is_integer_dtype(dtype):
            return True, int(v)
        if pd.api.types.is_float_dtype(dtype):
            return True, float(v)
        # For other numeric-like, try numeric
        if pd.api.types.is_numeric_dtype(dtype):
            return True, float(v)
        # For object/string dtype, keep as string
        return True, str(v)
    except Exception as e:
        return False, f"Could not convert '{v}' to {dtype} ({e})"

# Style function for missing highlight
def highlight_missing(col):
    return [f'background-color: {UI_BLUE}; color: {HIGHLIGHT_TEXT}; font-weight: bold;' if v>0 else '' for v in col]

# ------------------------- UI Layout -----------------------------
st.markdown("""
<div style='background:#0A81D1; color:white; padding:20px; border-radius:12px; margin-bottom:16px;'>
  <h1 style='margin:0'>Refined Data Profiling & Cleaning</h1>
  <p style='margin:0.1em 0 0 0'>Fast profiling, guided cleaning, and robust exports (CSV/Excel/PDF/HTML).</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("📤 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

if 'charts_meta' not in st.session_state:
    st.session_state.charts_meta = []

if not uploaded_file:
    st.info("⬆ Please upload a file to start.")
    st.stop()

# ------------------------- Data Ingestion ------------------------
try:
    if uploaded_file.name.endswith('.csv'):
        uploaded_file.seek(0)
        delimiter = detect_delimiter(uploaded_file)
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding='utf-8')
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

original_df = df.copy(deep=True)

# Basic cards
card1, card2, card3 = st.columns(3)
with card1:
    st.metric("Rows", f"{df.shape[0]}")
with card2:
    st.metric("Columns", f"{df.shape[1]}")
with card3:
    st.metric("Total Nulls", f"{int(df.isnull().sum().sum())}")

# Column summary
col_data = []
for col in df.columns:
    col_type = df[col].dtype
    sample = ", ".join(map(str, df[col].dropna().unique()[:5]))
    col_data.append([col, str(col_type), df[col].nunique(), int(df[col].isna().sum()), sample])
summary_df = pd.DataFrame(col_data, columns=["Column","Type","Unique","Missing","Sample Values"])

st.markdown("#### Data Summary")
st.dataframe(summary_df.style.apply(highlight_missing, subset=["Missing"]), use_container_width=True)

st.markdown("#### Data Example (Head & Tail)")
colH, colT = st.columns(2)
with colH:
    st.dataframe(df.head(5), use_container_width=True)
with colT:
    st.dataframe(df.tail(5), use_container_width=True)

# Prepare lists
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

profile_log = []
skipped_outlier_columns = []

# ---------------------- Outlier Detection ------------------------
st.markdown("<h3 style='color:{HEADING_COLOR};'>Step 2: Outlier Detection</h3>", unsafe_allow_html=True)

# Only numeric columns with >2 unique values are considered
eligible_outlier_cols = [c for c in numeric_cols if df[c].nunique(dropna=True)>2]
# Exclude binary-like columns that are numeric but only 0/1
eligible_outlier_cols = [c for c in eligible_outlier_cols if set(df[c].dropna().unique()) not in [set([0,1]), set([1,0])]]

UI_BLUE = "#0A81D1" 

for col in df.columns:
    # skip non-numeric and binary-like
    if col not in eligible_outlier_cols:
        skipped_outlier_columns.append(col)
        continue

    col_data_clean = df[col].dropna()
    q1, q3 = np.percentile(col_data_clean, [25,75])
    iqr = q3-q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    outliers = df[(df[col]<lower) | (df[col]>upper)][col]
    st.write(f"{col}: Outliers detected: {len(outliers)} of {df[col].dropna().shape[0]}")
    fig = px.box(df, y=col, title=f"Box: {col}", color_discrete_sequence=[UI_BLUE])
    st.plotly_chart(fig, use_container_width=True)
    # store metadata for PDF (histogram + box indicated as hist)
    st.session_state.charts_meta.append(("hist", f"Histogram: {col}", df[col]))

    # handling options
    st.markdown(
        f"<p style='color:#0A81D1; font-size:1.4em;'>How to handle outliers in <b>{col}</b>?</p>",
        unsafe_allow_html=True
    )
    choice = st.radio(
        "",  # no label here
        ["Keep all", "Remove", "Replace with median", "Replace with mean"],
        key=f"out_{col}"
    )

    if choice == "Remove":
        before = df.shape[0]
        df = df[~((df[col]<lower) | (df[col]>upper)) | df[col].isna()]
        removed = before - df.shape[0]
        st.success(f"Removed {removed} rows with outliers in {col}")
        profile_log.append(f"Outliers removed in {col}: {removed}")
    elif choice == "Replace with median":
        med = df[col].median()
        df.loc[((df[col]<lower) | (df[col]>upper)), col] = med
        st.info(f"Outliers replaced with median ({med}) in {col}")
        profile_log.append(f"Outliers replaced with median in {col}")
    elif choice == "Replace with mean":
        m = df[col].mean()
        df.loc[((df[col]<lower) | (df[col]>upper)), col] = m
        st.info(f"Outliers replaced with mean ({m}) in {col}")
        profile_log.append(f"Outliers replaced with mean in {col}")
    else:
        st.info("No outlier treatment applied.")
        profile_log.append(f"Outliers kept in {col}")

# After all columns
# Detect categorical (non-numeric) columns
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Detect binary columns (numeric or string) with exactly 2 unique non-null values
binary_cols = [col for col in df.columns if df[col].nunique(dropna=True) == 2]

# Combine and deduplicate
skipped_cols = sorted(list(set(non_numeric_cols + binary_cols)))

if skipped_cols:
    # Arrange into two columns
    col_pairs = [f"{skipped_cols[i]:<30} {skipped_cols[i+1] if i+1 < len(skipped_cols) else ''}"
                 for i in range(0, len(skipped_cols), 2)]
    formatted_cols = "\n".join(col_pairs)

    st.info(
        f"No outlier handling was needed for the following columns "
        f"(categorical or binary types):\n\n\n{formatted_cols}\n"
    )


# ----------------------- Null Value Handling ---------------------
st.markdown("<h3 style='color:{HEADING_COLOR};'>Step 3: Null Value Handling</h3>", unsafe_allow_html=True)
missing_summary = []
for col in df.columns:
    miss = int(df[col].isna().sum())
    if miss==0:
        continue
    pct = 100*miss/len(df)
    missing_summary.append([col, miss, f"{pct:.2f}%"])

if missing_summary:
    st.dataframe(pd.DataFrame(missing_summary, columns=["Column","Nulls","Percentage"]), use_container_width=True)
    for col, miss, pct in missing_summary:
        col_dtype = df[col].dtype
        st.markdown(
            f"<p style='color:#0A81D1; font-size:1.4em;'>Null handling for <b>{col}</b> ({col_dtype})</p>",
            unsafe_allow_html=True
        )

        if pd.api.types.is_numeric_dtype(col_dtype):
            methods = ["Fill with mean","Fill with median","Fill with custom value","Drop rows with nulls"]
        else:
            methods = ["Fill with mode","Fill with 'Unknown'","Fill with custom value","Drop rows with nulls"]
        method = st.selectbox(f"Null handling for {col}", options=methods, key=f"null_{col}")
        if method == "Fill with mean":
            val = df[col].mean()
            df[col] = df[col].fillna(val)
            profile_log.append(f"Filled nulls in {col} with mean ({val})")
            st.info(f"Filled nulls in {col} with mean ({val})")
        elif method == "Fill with median":
            val = df[col].median()
            df[col] = df[col].fillna(val)
            profile_log.append(f"Filled nulls in {col} with median ({val})")
            st.info(f"Filled nulls in {col} with median ({val})")
        elif method == "Fill with mode":
            val = df[col].mode().iloc[0]
            df[col] = df[col].fillna(val)
            profile_log.append(f"Filled nulls in {col} with mode ({val})")
            st.info(f"Filled nulls in {col} with mode ({val})")
        elif method == "Fill with 'Unknown'":
            df[col] = df[col].fillna("Unknown")
            profile_log.append(f"Filled nulls in {col} with 'Unknown'")
            st.info(f"Filled nulls in {col} with 'Unknown'")
        elif method == "Fill with custom value":
            custom = st.text_input(f"Custom fill value for {col}", key=f"cust_{col}")
            if custom != "":
                ok, res = try_cast_fill(custom, col_dtype)
                if ok:
                    df[col] = df[col].fillna(res)
                    profile_log.append(f"Filled nulls in {col} with custom value ({res})")
                    st.info(f"Filled nulls in {col} with custom value ({res})")
                else:
                    st.error(res)
        else:
            before = df.shape[0]
            df = df[df[col].notna()]
            dropped = before - df.shape[0]
            profile_log.append(f"Dropped {dropped} rows with nulls in {col}")
            st.warning(f"Dropped {dropped} rows with nulls in {col}")
else:
    st.success("No nulls detected.")

# ---------------------- Duplicate Handling -----------------------
st.markdown("<h3 style='color:{HEADING_COLOR};'>Step 4: Duplicate Handling</h3>", unsafe_allow_html=True)
dupes = original_df.duplicated(keep='first')
num_dupes = int(dupes.sum())
if num_dupes>0:
    dupe_examples = original_df[dupes].head(5)
    st.warning(f"Found {num_dupes} duplicate rows. They will be removed automatically. Showing up to 5 examples:")
    st.dataframe(dupe_examples)
    profile_log.append(f"Duplicates found: {num_dupes}. Examples saved to report.")
    # remove duplicates from df
    df = df.drop_duplicates()
    profile_log.append(f"Duplicates removed: {num_dupes}")
else:
    st.info("No duplicates found.")

# ------------------------- Visualizations -----------------------
st.subheader("📊 Visualizations")
# Reset charts_meta for this dataset
st.session_state.charts_meta = []
UI_BLUE = "#0A81D1"

# Null bar
nulls_after = [int(df[c].isnull().sum()) for c in df.columns]
fig_nulls = px.bar(x=df.columns, y=nulls_after, labels={'x':'Column','y':'Nulls'}, title='Null values after cleaning', color_discrete_sequence=[UI_BLUE])
st.plotly_chart(fig_nulls, use_container_width=True)
st.session_state.charts_meta.append(("bar", "Null values after cleaning", pd.Series(dict(zip(df.columns, nulls_after)))))

# Random numeric histograms
random_numeric = list(df.select_dtypes(include=np.number).columns)
random_numeric = random_numeric[:5]
for col in random_numeric:
    fig_hist = px.histogram(df, x=col, nbins=30, title=f"Histogram: {col}", marginal='box', color_discrete_sequence=[UI_BLUE])
    st.plotly_chart(fig_hist, use_container_width=True)
    st.session_state.charts_meta.append(("hist", f"Histogram: {col}", df[col]))

# Top categories pie/bar
random_categorical = list(df.select_dtypes(include='object').columns)[:5]
for col in random_categorical:
    vc = df[col].value_counts().head(10)
    fig_pie = px.pie(values=vc.values, names=vc.index, title=f"Distribution: {col}")
    st.plotly_chart(fig_pie, use_container_width=True)
    st.session_state.charts_meta.append(("pie", f"Distribution: {col}", df[col]))

# Correlation heatmap
numeric_cols_now = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols_now) > 1:
    fig_corr = px.imshow(df[numeric_cols_now].corr(), text_auto=True, title='Correlation Heatmap')
    st.plotly_chart(fig_corr, use_container_width=True)
    st.session_state.charts_meta.append(("corr", "Correlation Heatmap", df[numeric_cols_now]))

# Scatter pairs - pick up to 3 pairs
pairs = []
if len(numeric_cols_now) > 1:
    for i in range(min(3, len(numeric_cols_now)-1)):
        pairs.append((numeric_cols_now[i], numeric_cols_now[i+1]))

for x_col, y_col in pairs:
    fig_sc = px.scatter(df, x=x_col, y=y_col, title=f"Scatter: {y_col} vs {x_col}")
    st.plotly_chart(fig_sc, use_container_width=True)
    st.session_state.charts_meta.append(("scatter", f"Scatter: {y_col} vs {x_col}", df[[x_col, y_col]]))

# Bar charts for categorical top categories
for col in random_categorical:
    vc = df[col].value_counts().head(10)
    fig_bar = px.bar(vc, title=f"Top Categories in {col}", color_discrete_sequence=[UI_BLUE])
    st.plotly_chart(fig_bar, use_container_width=True)
    st.session_state.charts_meta.append(("bar", f"Top Categories in {col}", df[col]))

# ------------------------- Summary & Export ---------------------
st.markdown("<h3 style='color:{HEADING_COLOR};'>Summary & Exportable Report</h3>", unsafe_allow_html=True)
report_txt = f"""
Basic Info
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {', '.join([f"{c} ({str(df[c].dtype)})" for c in df.columns])}
- Unique value counts: {[int(df[c].nunique()) for c in df.columns]}

Cleaning Summary
"""
report_txt += '\n'.join(["- "+s for s in profile_log])
st.markdown(report_txt)

# CSV Download
st.download_button("⬇ Download Cleaned CSV", df.to_csv(index=False), "cleaned_data.csv")
# Excel Download
excel_buffer = io.BytesIO()
df.to_excel(excel_buffer, index=False, engine='openpyxl')
excel_buffer.seek(0)
st.download_button(label="📒 Download Cleaned Excel", data=excel_buffer, file_name="cleaned_data.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# PDF generation using Matplotlib charts stored in charts_meta
if st.button("📄 Generate PDF Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Add header text
    for line in report_txt.strip().split('\n'):
        pdf.multi_cell(0, 8, txt=line)
    # Add duplicates info if any
    if num_dupes>0:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, f"Duplicates found: {num_dupes}", ln=True)
    # Export charts
    for chart_type, title, data in st.session_state.charts_meta:
        try:
            if chart_type == 'corr':
                add_matplotlib_correlation_heatmap(data, pdf, title)
            else:
                add_matplotlib_chart_to_pdf(data, chart_type, pdf, title)
        except Exception as e:
            st.warning(f"Could not add chart '{title}' to PDF: {e}")
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name='data_cleaning_report.pdf')

# HTML report
with st.expander("🖥 Export Complete HTML Report"):
    st.markdown(report_txt + "\n\n" + "\n".join([f"- {s}" for s in profile_log]), unsafe_allow_html=True)

# ------------------------- Insights & Conclusion ----------------
st.header("🔑 Insights")
for col in df.select_dtypes(include=np.number).columns:
    avg = df[col].mean()
    st.markdown(f"Average *{col}*: <span style='color:{UI_BLUE}; font-weight:600'>{avg:,.2f}</span>", unsafe_allow_html=True)

for col in df.select_dtypes(include='object').columns:
    if df[col].dropna().shape[0]>0:
        top_cat = df[col].mode()[0]
        st.markdown(f"Top category in {col}: <span style='color:#29A746;font-weight:600'>{top_cat}</span>", unsafe_allow_html=True)

st.success("Conclusion: Data is now ready for statistical analysis!")

# ---------------------------- End --------------------------------
st.markdown("<div style='text-align:center;padding:24px 0 6px;'><img src='https://static.streamlit.io/examples/dice.jpg' width=55 /><br><span style='font-size:1.2em;font-weight:500;color:#0A81D1'>Thanks for using the Refined Data Profiling & Cleaning App!</span></div>", unsafe_allow_html=True)