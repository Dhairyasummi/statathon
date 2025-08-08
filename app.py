import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io
from fpdf import FPDF
from PIL import Image


################
# Step 1: Data Ingestion & Initial Summary
################

st.title("Refined Data Profiling & Cleaning App")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

def detect_delimiter(file_bytes):
    try:
        sample = file_bytes.read(5000).decode('utf-8')
        dlms = [',', ';', '\t', '|']
        counts = [sample.count(d) for d in dlms]
        return dlms[np.argmax(counts)]
    except:
        return ','

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        delimiter = detect_delimiter(uploaded_file)
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding='utf-8')
    else:
        df = pd.read_excel(uploaded_file)
        
    st.subheader("Initial Data Profile")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
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
    st.dataframe(summary_df)

    st.write("**Example (Head)**")
    st.dataframe(df.head(5))
    st.write("**Example (Tail)**")
    st.dataframe(df.tail(5))

    shape_txt = f"{df.shape[0]} rows × {df.shape[1]} columns"
else:
    st.info("Awaiting file upload...")
    st.stop()

profile_log = []
input_log = {}

################
# Step 2: Outlier Detection
################

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
outlier_info = {}
outlier_graphs = {}
st.subheader("Step 2: Outlier Detection")
st.write("Each numeric column is checked for outliers. Remove or replace?")

for col in numeric_cols:
    col_data = df[col].dropna()
    q1, q3 = np.percentile(col_data, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = col_data[(col_data < lower) | (col_data > upper)]
    
    outlier_info[col] = {
        "count": len(outliers),
        "method": "IQR",
        "bounds": (lower, upper),
        "total": len(col_data),
    }
    
    fig = px.box(df, y=col, title=f"{col} (Outliers highlighted)")
    outlier_graphs[col] = fig
    
    st.write(f"**{col}**: Outliers detected: {len(outliers)} out of {len(col_data)}")
    st.plotly_chart(fig, use_container_width=True)
    
    explain = st.expander("Why?", expanded=False)
    with explain:
        st.markdown(f"Outliers are defined as values < {lower:.2f} or > {upper:.2f} (using IQR method).")

    choice = st.radio(
        f"Remove outliers in {col}?",
        ["Keep all", "Remove", "Replace with median", "Replace with mean"],
        key=f"outlier_{col}"
    )
    input_log[f"outlier_{col}"] = choice
    if choice == "Remove":
        df = df[(df[col] >= lower) & (df[col] <= upper) | df[col].isna()]
        profile_log.append(f"Outliers removed in {col}: {len(outliers)}")
    elif choice == "Replace with median":
        median_val = col_data.median()
        df[col] = np.where((df[col] < lower) | (df[col] > upper), median_val, df[col])
        profile_log.append(f"Outliers replaced (median) in {col}: {len(outliers)}")
    elif choice == "Replace with mean":
        mean_val = col_data.mean()
        df[col] = np.where((df[col] < lower) | (df[col] > upper), mean_val, df[col])
        profile_log.append(f"Outliers replaced (mean) in {col}: {len(outliers)}")
    else:
        profile_log.append(f"Outliers kept in {col}")

################
# Step 3: Null Value Handling
################

st.subheader("Step 3: Null Value Handling")
missing_summary = []
for col in df.columns:
    missing = df[col].isna().sum()
    if missing == 0:
        continue
    missing_pct = 100 * missing / len(df)
    missing_summary.append([col, missing, f"{missing_pct:.2f}%"])

if missing_summary:
    st.write(pd.DataFrame(missing_summary, columns=["Column", "Nulls", "Percentage"]))
    for col, missing, pct in missing_summary:
        col_type = df[col].dtype
        if col_type in [np.float64, np.int64]:
            methods = ["Fill with mean", "Fill with median", "Fill with custom value", "Drop rows with nulls"]
        else:
            methods = ["Fill with mode", "Fill with 'Unknown'", "Fill with custom value", "Drop rows with nulls"]

        method = st.selectbox(f"Null handling for {col}", options=methods, key=f"null_{col}")
        input_log[f"null_{col}"] = method

        if "mean" in method:
            fill = df[col].mean()
            df[col] = df[col].fillna(fill)
            profile_log.append(f"Filled nulls in {col} with mean ({fill:.2f})")
        elif "median" in method:
            fill = df[col].median()
            df[col] = df[col].fillna(fill)
            profile_log.append(f"Filled nulls in {col} with median ({fill:.2f})")
        elif "mode" in method:
            fill = df[col].mode().iloc[0]
            df[col] = df[col].fillna(fill)
            profile_log.append(f"Filled nulls in {col} with mode ({fill})")
        elif "'Unknown'" in method:
            df[col] = df[col].fillna("Unknown")
            profile_log.append(f"Filled nulls in {col} with 'Unknown'")
        elif "custom" in method:
            fill = st.text_input(f"Custom fill value for {col}", key=f"custom_{col}")
            if fill != "":
                df[col] = df[col].fillna(fill)
                profile_log.append(f"Filled nulls in {col} with custom value ({fill})")
        elif "Drop" in method:
            df = df[df[col].notna()]
            profile_log.append(f"Dropped rows with nulls in {col}")
else:
    st.info("No nulls detected.")

################
# Step 4: Duplicate Handling (Automated)
################

st.subheader("Step 4: Duplicate Handling")
dupes = df.duplicated().sum()
pct_dupes = 100 * dupes / len(df)
if dupes:
    df = df.drop_duplicates()
    profile_log.append(f"Duplicates removed: {dupes} ({pct_dupes:.2f}% of total)")
    st.success(f"Removed {dupes} duplicates ({pct_dupes:.2f}%)")
else:
    st.info("No duplicates found.")

################
# Step 5: Output & Report Generation
################

st.header("Summary & Exportable Report")
report_txt = f"""
**Basic Info**
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {', '.join([f"{c} ({str(df[c].dtype)})" for c in df.columns])}
- Unique value counts: {[df[c].nunique() for c in df.columns]}

**Cleaning Summary**
"""
report_txt += '\n'.join(["- " + log for log in profile_log])
st.markdown(report_txt)

# Visuals
st.subheader("Visualizations")
nulls_after = [df[c].isnull().sum() for c in df.columns]
fig1 = px.bar(
    x=df.columns,
    y=nulls_after,
    labels={'x': 'Column', 'y': 'Nulls'},
    title="Null values after cleaning"
)
st.plotly_chart(fig1)

for col in numeric_cols:
    fig = px.box(df, y=col, title=f"{col} (After Outlier Handling)")
    st.plotly_chart(fig)

cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    vc = df[col].value_counts().head(10)
    fig = px.bar(vc, title=f"Distribution: {col}")
    st.plotly_chart(fig)

################
# Export Option
################


st.subheader("Download Cleaned Data & Report")

# CSV Download
st.download_button("Download Cleaned CSV", df.to_csv(index=False), "cleaned_data.csv")

# Excel Download (corrected!)
excel_buffer = io.BytesIO()
df.to_excel(excel_buffer, index=False, engine='openpyxl')
excel_buffer.seek(0)
st.download_button(
    label="Download Cleaned Excel",
    data=excel_buffer,
    file_name="cleaned_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


# PDF Report

# Helper function to add Plotly figure to PDF
# PDF Report

# Helper function to add Plotly figure to PDF
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

# Collect visualizations already shown in Streamlit
charts_to_include = []

# Add nulls bar chart
charts_to_include.append(("Null Values After Cleaning", fig1))

# Add numeric column boxplots (already shown above)
for col in numeric_cols:
    fig = px.box(df, y=col, title=f"{col} (After Outlier Handling)")
    charts_to_include.append((f"Boxplot: {col}", fig))

# Add categorical bar plots (already shown above)
for col in cat_cols:
    vc = df[col].value_counts().head(10)
    fig = px.bar(vc, title=f"Top Categories in {col}")
    charts_to_include.append((f"Category Distribution: {col}", fig))

# Generate PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# Add report text
for line in report_txt.strip().split('\n'):
    pdf.multi_cell(0, 10, txt=line)

# Add all collected plots to the PDF
for title, fig in charts_to_include:
    add_plot_to_pdf(fig, pdf, title=title)

# Save to byte stream and download
pdf_bytes = pdf.output(dest="S").encode('latin1')
st.download_button("Download PDF Report", data=pdf_bytes, file_name="data_cleaning_report.pdf")



# Optional HTML report
with st.expander("Export Complete HTML Report"):
    st.markdown(report_txt, unsafe_allow_html=True)

################
# Insights & Conclusion
################

st.header("Insights")
for col in numeric_cols:
    avg = df[col].mean()
    st.write(f"Average {col}: {avg:,.2f}")
for col in cat_cols:
    top_cat = df[col].mode()[0]
    st.write(f"Top category in {col}: {top_cat}")

st.success("**Conclusion:** Data is now ready for statistical analysis.")
