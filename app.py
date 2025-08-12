import random
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from fpdf import FPDF
from PIL import Image

# -------------- Custom Theming (Create .streamlit/config.toml for persistent theming) -----------
st.set_page_config(page_title="Refined Data Profiling & Cleaning", layout="wide")

# -------------- Custom Header Banner -----------
st.markdown("""
<div style='
     background:#0A81D1;
     color:white;
     padding:28px 18px 12px 18px;
     border-radius:16px;
     margin-bottom:24px;'
>
    <h1 style='margin-bottom:0;'>Refined Data Profiling & Cleaning App</h1>
    <p style='font-size:1.1em;'>Easy profiling, outlier handling, null checks, and clean report export in one place.</p>
</div>
""", unsafe_allow_html=True)

# -------------- Sidebar Upload -----------
st.sidebar.header("📤 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

def detect_delimiter(file_bytes):
    try:
        sample = file_bytes.read(5000).decode('utf-8')
        dlms = [',', ';', '\t', '|']
        counts = [sample.count(d) for d in dlms]
        return dlms[np.argmax(counts)]
    except:
        return ','

# -------------- Section: Data Ingestion & Initial Summary -----------
st.markdown(
    "<h3 style='color: #0A81D1; border-left: 5px solid #4FC3F7; padding-left: 10px;'>Step 1: Data Ingestion & Initial Summary</h3>", 
    unsafe_allow_html=True
)

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        delimiter = detect_delimiter(uploaded_file)
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding='utf-8')
    else:
        df = pd.read_excel(uploaded_file)

    # Card Metric Display
    card1, card2, card3 = st.columns(3)
    with card1:
        st.metric("Rows", f"{df.shape[0]}")
    with card2:
        st.metric("Columns", f"{df.shape[1]}")
    with card3:
        st.metric("Total Nulls", int(df.isnull().sum().sum()))

    # Column summary table with conditional formatting
    col_data = []
    for col in df.columns:
        col_type = df[col].dtype
        unique_vals_sample = ", ".join(map(str, df[col].unique()[:5]))
        col_data.append([
            col,
            str(col_type),
            df[col].nunique(),
            df[col].isna().sum(),
            unique_vals_sample
        ])
    summary_df = pd.DataFrame(
        col_data, columns=["Column", "Type", "Unique", "Missing", "Sample Values"]
    )

    def highlight_missing(s):
        bg = ['background-color: #FFE1E0' if v > 0 else '' for v in s]
        return bg

    st.markdown("#### Data Summary")
    st.dataframe(
        summary_df.style.apply(highlight_missing, subset=["Missing"]),
        use_container_width=True
    )

    st.write("#### Data Example (Head & Tail)")
    colH, colT = st.columns(2)
    with colH:
        st.dataframe(df.head(5), use_container_width=True)
    with colT:
        st.dataframe(df.tail(5), use_container_width=True)

    shape_txt = f"{df.shape[0]} rows × {df.shape[1]} columns"

    # Initialize list to store chart metadata for PDF export
    charts_meta = []
else:
    st.info("⬆ Please upload a file to start.")
    st.stop()

profile_log = []
input_log = {}

# -------------- Section: Outlier Detection -----------
st.markdown(
    "<h3 style='color: #29A746; border-left: 5px solid #80FFB0; padding-left: 10px;'>Step 2: Outlier Detection</h3>", 
    unsafe_allow_html=True
)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
outlier_info = {}
outlier_graphs = {}

for col in numeric_cols:
    col_data_clean = df[col].dropna()
    q1, q3 = np.percentile(col_data_clean, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = col_data_clean[(col_data_clean < lower) | (col_data_clean > upper)]

    outlier_info[col] = {
        "count": len(outliers),
        "method": "IQR",
        "bounds": (lower, upper),
        "total": len(col_data_clean)
    }

    fig = px.box(df, y=col, boxmode="overlay", color_discrete_sequence=["#0A81D1"])

    outlier_graphs[col] = fig

    st.write(f"{col}: Outliers detected: {len(outliers)} of {len(col_data_clean)}")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ Why are these outliers?", expanded=False):
        st.markdown(f"Outliers defined as values < *{lower:.2f}* or > *{upper:.2f}* (IQR method).")
    
    st.markdown("<div style='background-color:#D2F7E6;padding:11px 7px;border-radius:6px;'>", unsafe_allow_html=True)
    choice = st.radio(
        f"How to handle outliers in {col}?",
        ["Keep all", "Remove", "Replace with median", "Replace with mean"],
        key=f"out_{col}"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    input_log[f"out_{col}"] = choice

    # Outlier handling logic
    if choice == "Remove":
        df = df[(df[col] >= lower) & (df[col] <= upper) | df[col].isna()]
        st.success(f"✅ Outliers removed in {col}: {len(outliers)}")
        profile_log.append(f"Outliers removed in {col}: {len(outliers)}")
    elif choice == "Replace with median":
        med_val = col_data_clean.median()
        df[col] = np.where((df[col] < lower) | (df[col] > upper), med_val, df[col])
        st.info(f"🛠 Outliers replaced with median ({med_val:.2f})")
        profile_log.append(f"Outliers replaced (median) in {col}: {len(outliers)}")
    elif choice == "Replace with mean":
        mean_val = col_data_clean.mean()
        df[col] = np.where((df[col] < lower) | (df[col] > upper), mean_val, df[col])
        st.info(f"🛠 Outliers replaced with mean ({mean_val:.2f})")
        profile_log.append(f"Outliers replaced (mean) in {col}: {len(outliers)}")
    else:
        st.info("No outlier treatment applied.")
        profile_log.append(f"Outliers kept in {col}")

# -------------- Section: Null Value Handling -----------

st.markdown(
    "<h3 style='color: #C1850A; border-left: 5px solid #FDD771; padding-left: 10px;'>Step 3: Null Value Handling</h3>", 
    unsafe_allow_html=True
)
missing_summary = []
for col in df.columns:
    missing = df[col].isna().sum()
    if missing == 0:
        continue
    missing_pct = 100 * missing / len(df)
    missing_summary.append([col, missing, f"{missing_pct:.2f}%"])

if missing_summary:
    st.dataframe(pd.DataFrame(missing_summary, columns=["Column", "Nulls", "Percentage"]), use_container_width=True)
    for col, missing, pct in missing_summary:
        col_type = df[col].dtype
        if col_type in [np.float64, np.int64]:
            methods = ["Fill with mean", "Fill with median", "Fill with custom value", "Drop rows with nulls"]
        else:
            methods = ["Fill with mode", "Fill with 'Unknown'", "Fill with custom value", "Drop rows with nulls"]

        st.markdown("<div style='background-color:#FFF4DC;padding:11px 7px;border-radius:6px;'>", unsafe_allow_html=True)
        method = st.selectbox(f"Null handling for {col}", options=methods, key=f"null_{col}")
        st.markdown("</div>", unsafe_allow_html=True)
        input_log[f"null_{col}"] = method

        # Null handling
        if "mean" in method:
            fill = df[col].mean()
            df[col] = df[col].fillna(fill)
            st.info(f"🧮 Filled nulls in {col} with mean ({fill:.2f})")
            profile_log.append(f"Filled nulls in {col} with mean ({fill:.2f})")
        elif "median" in method:
            fill = df[col].median()
            df[col] = df[col].fillna(fill)
            st.info(f"🧮 Filled nulls in {col} with median ({fill:.2f})")
            profile_log.append(f"Filled nulls in {col} with median ({fill:.2f})")
        elif "mode" in method:
            fill = df[col].mode().iloc[0]
            df[col] = df[col].fillna(fill)
            st.info(f"🧮 Filled nulls in {col} with mode ({fill})")
            profile_log.append(f"Filled nulls in {col} with mode ({fill})")
        elif "'Unknown'" in method:
            df[col] = df[col].fillna("Unknown")
            st.info(f"🏷 Filled nulls in {col} with 'Unknown'")
            profile_log.append(f"Filled nulls in {col} with 'Unknown'")
        elif "custom" in method:
            fill = st.text_input(f"Custom fill value for {col}", key=f"custom_{col}")
            if fill != "":
                df[col] = df[col].fillna(fill)
                st.info(f"🧑‍💻 Filled nulls in {col} with custom value ({fill})")
                profile_log.append(f"Filled nulls in {col} with custom value ({fill})")
        elif "Drop" in method:
            df = df[df[col].notna()]
            st.warning(f"🧹 Dropped rows with nulls in {col}")
            profile_log.append(f"Dropped rows with nulls in {col}")
else:
    st.success("No nulls detected.")

# -------------- Section: Duplicate Handling -----------
st.markdown(
    "<h3 style='color: #8B24C4; border-left: 5px solid #E3B7F6; padding-left: 10px;'>Step 4: Duplicate Handling</h3>", 
    unsafe_allow_html=True
)
dupes = df.duplicated().sum()
pct_dupes = 100 * dupes / len(df)
if dupes:
    df = df.drop_duplicates()
    st.success(f"✅ Removed {dupes} duplicates ({pct_dupes:.2f}%)")
    profile_log.append(f"Duplicates removed: {dupes} ({pct_dupes:.2f}% of total)")
else:
    st.info("✨ No duplicates found!")

# -------------- Section: Output & Report Generation -----------
st.markdown(
    "<h3 style='color: #0A81D1; border-left: 5px solid #4FC3F7; padding-left: 10px;'>Summary & Exportable Report</h3>", 
    unsafe_allow_html=True
)
report_txt = f"""
*Basic Info*
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {', '.join([f"{c} ({str(df[c].dtype)})" for c in df.columns])}
- Unique value counts: {[df[c].nunique() for c in df.columns]}

*Cleaning Summary*
"""
report_txt += '\n'.join(["- " + log for log in profile_log])
st.markdown(report_txt)


# === Prepare consistent column lists and one-time random selection ===
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

# Initialize one-time random picks per dataset upload (persist in session_state)
if 'random_numeric' not in st.session_state:
    st.session_state.random_numeric = random.sample(numeric_cols, min(5, len(numeric_cols))) if numeric_cols else []
    st.session_state.random_categorical = random.sample(cat_cols, min(5, len(cat_cols))) if cat_cols else []
    # pairs from the numeric selection
    numeric_pairs_all = [(x, y) for i, x in enumerate(st.session_state.random_numeric) for y in st.session_state.random_numeric[i+1:]]
    st.session_state.random_pairs = random.sample(numeric_pairs_all, min(3, len(numeric_pairs_all))) if numeric_pairs_all else []

random_numeric = st.session_state.random_numeric
random_categorical = st.session_state.random_categorical
random_pairs = st.session_state.random_pairs

# Placeholder for fig_nulls to guarantee it exists for PDF export even if visuals section aborts
fig_nulls = None

# --- Visualizations ---
st.subheader("📊 Visualizations")

# Null values bar chart (define fig_nulls consistently)
nulls_after = [df[c].isnull().sum() for c in df.columns]
fig_nulls = px.bar(
    x=df.columns,
    y=nulls_after,
    labels={'x': 'Column', 'y': 'Nulls'},
    title="Null values after cleaning",
    color_discrete_sequence=["#F05050"]
)
st.plotly_chart(fig_nulls, use_container_width=True)

# Histograms for randomly selected numeric columns
for col in random_numeric:
    # skip if column not present (safety)
    if col not in df.columns:
        continue
    fig_hist = px.histogram(df, x=col, nbins=30, title=f"Histogram: {col}", marginal="box")
    st.plotly_chart(fig_hist, use_container_width=True)
charts_meta.append(("hist", f"Histogram: {col}", df[col]))

# Pie charts for randomly selected categorical columns (only if cardinality reasonable)
for col in random_categorical:
    if col not in df.columns:
        continue
    vc = df[col].value_counts().head(10)
    fig_pie = px.pie(values=vc.values, names=vc.index, title=f"Distribution: {col}")
    st.plotly_chart(fig_pie, use_container_width=True)
charts_meta.append(("pie", f"Distribution: {col}", df[col]))

# Correlation heatmap for all numeric columns (overall)
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)
charts_meta.append(("corr", "Correlation Heatmap", df[numeric_cols]))

# Scatter plots for random numeric pairs
for x_col, y_col in random_pairs:
    if x_col in df.columns and y_col in df.columns:
        fig_scatter = px.scatter(df, x=x_col, y=y_col, title=f"Scatter: {y_col} vs {x_col}")
        st.plotly_chart(fig_scatter, use_container_width=True)
charts_meta.append(("scatter", f"Scatter: {y_col} vs {x_col}", df[[x_col, y_col]]))

# Bar charts for randomly selected categorical columns (top categories)
for col in random_categorical:
    if col not in df.columns:
        continue
    vc = df[col].value_counts().head(10)
    fig_bar = px.bar(vc, title=f"Top Categories in {col}", color_discrete_sequence=["#0A81D1"])
    st.plotly_chart(fig_bar, use_container_width=True)
charts_meta.append(("bar", f"Top Categories in {col}", df[col]))
# ------------- Download Section -------------
st.subheader("⬇ Download Cleaned Data & Report")
# CSV Download
st.download_button("⬇ Download Cleaned CSV", df.to_csv(index=False), "cleaned_data.csv")
# Excel Download
excel_buffer = io.BytesIO()
df.to_excel(excel_buffer, index=False, engine='openpyxl')
excel_buffer.seek(0)
st.download_button(
    label="📒 Download Cleaned Excel",
    data=excel_buffer,
    file_name="cleaned_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ----------- PDF Report with Plots Support -----------
def add_plot_to_pdf(fig, pdf, title=None):
    try:
        import plotly.io as pio
        buf = io.BytesIO()
        fig.write_image(buf, format='png')  # Requires kaleido
        buf.seek(0)
        img = Image.open(buf)
        img_path = "temp_plot.png"
        img.save(img_path)

        if title:
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, title, ln=True)
        pdf.image(img_path, w=180)
        pdf.ln(5)
    except Exception as e:
        st.warning(f"Could not add plot '{title}' to PDF: {e}")
        pdf.set_font("Arial", 'I', 10)
        pdf.ln(5)
        pdf.cell(0, 10, f"[Plot '{title}' could not be included due to export error.]", ln=True)



# =============================
# Matplotlib PDF Export Helpers
# =============================
import matplotlib.pyplot as plt
import seaborn as sns

def add_matplotlib_chart_to_pdf(data, chart_type, pdf, title):
    fig, ax = plt.subplots()
    if chart_type == "hist":
        ax.hist(data.dropna(), bins=20, color="#0A81D1")
        ax.set_xlabel(data.name if hasattr(data, 'name') else 'Value')
        ax.set_ylabel("Frequency")
    elif chart_type == "pie":
        counts = data.value_counts()
        ax.pie(counts, labels=counts.index.astype(str), autopct='%1.1f%%')
    elif chart_type == "bar":
        counts = data.value_counts()
        ax.bar(counts.index.astype(str), counts.values, color="#0A81D1")
        ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")
    elif chart_type == "scatter":
        if isinstance(data, pd.DataFrame) and len(data.columns) >= 2:
            ax.scatter(data.iloc[:, 0], data.iloc[:, 1], alpha=0.7)
            ax.set_xlabel(data.columns[0])
            ax.set_ylabel(data.columns[1])
    ax.set_title(title)
    img_path = "temp_plot.png"
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches="tight")
    plt.close(fig)
    pdf.image(img_path, w=180)

def add_matplotlib_correlation_heatmap(df, pdf, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title(title)
    img_path = "temp_corr.png"
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches="tight")
    plt.close(fig)
    pdf.image(img_path, w=180)

# Generate PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# Add report text (line by line for formatting)
for line in report_txt.strip().split('\n'):
    pdf.multi_cell(0, 10, txt=line)

# Add all collected plots to PDF using Matplotlib fallback
for chart_type, title, data in charts_meta:
    if chart_type == "corr":
        add_matplotlib_correlation_heatmap(data, pdf, title)
    else:
        add_matplotlib_chart_to_pdf(data, chart_type, pdf, title)

# Save to bytes for Streamlit download
pdf_bytes = pdf.output(dest="S").encode('latin1')
st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="data_cleaning_report.pdf")

# Optional: HTML report for copying/sharing
with st.expander("🖥 Export Complete HTML Report"):
    st.markdown(report_txt, unsafe_allow_html=True)

# ------------- Insights & Conclusion -------------
st.header("🔑 Insights")
for col in numeric_cols:
    avg = df[col].mean()
    st.write(f"Average {col}: <span style='color:#0A81D1;font-weight:600'>{avg:,.2f}</span>", unsafe_allow_html=True)
for col in cat_cols:
    top_cat = df[col].mode()[0]
    st.write(f"Top category in *{col}*: <span style='color:#29A746;font-weight:600'>{top_cat}</span>", unsafe_allow_html=True)

st.success("*Conclusion:* Data is now ready for statistical analysis!")

# ------------- End Banner -------------
st.markdown(
    "<div style='text-align:center;padding:24px 0 6px;'><img src='https://static.streamlit.io/examples/dice.jpg' width=55 /><br><span style='font-size:1.2em;font-weight:500;color:#0A81D1'>Thanks for using the Refined Data Profiling & Cleaning App!</span></div>",
    unsafe_allow_html=True
)

# =============================
# Matplotlib PDF Export Helpers
# =============================
import matplotlib.pyplot as plt
import seaborn as sns

def add_matplotlib_chart_to_pdf(data, chart_type, pdf, title):
    fig, ax = plt.subplots()
    if chart_type == "hist":
        ax.hist(data.dropna(), bins=20, color="#0A81D1")
        ax.set_xlabel(data.name if hasattr(data, 'name') else 'Value')
        ax.set_ylabel("Frequency")
    elif chart_type == "pie":
        counts = data.value_counts()
        ax.pie(counts, labels=counts.index.astype(str), autopct='%1.1f%%')
    elif chart_type == "bar":
        counts = data.value_counts()
        ax.bar(counts.index.astype(str), counts.values, color="#0A81D1")
        ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")
    elif chart_type == "scatter":
        if isinstance(data, pd.DataFrame) and len(data.columns) >= 2:
            ax.scatter(data.iloc[:, 0], data.iloc[:, 1], alpha=0.7)
            ax.set_xlabel(data.columns[0])
            ax.set_ylabel(data.columns[1])
    ax.set_title(title)
    img_path = "temp_plot.png"
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches="tight")
    plt.close(fig)
    pdf.image(img_path, w=180)

def add_matplotlib_correlation_heatmap(df, pdf, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title(title)
    img_path = "temp_corr.png"
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches="tight")
    plt.close(fig)
    pdf.image(img_path, w=180)
