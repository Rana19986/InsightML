# ============================================================
# InsightML — ADAPTIVE MACHINE LEARNING & DATA ANALYTICS
# Version 1.0 | Google Colab | Free
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import gradio as gr

from pathlib import Path

# ------------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("InsightML dashboard code loaded.")

# ------------------------------------------------------------
# 2. ANALYSIS ENGINE
# ------------------------------------------------------------

def analyze_dataset(file_path, target_column):

    if file_path is None:
        raise gr.Error("Please upload a CSV file first.")

    try:
        # Load uploaded data.
        df = pd.read_csv(file_path, low_memory=False)

    except Exception as exc:
        raise gr.Error(f"Could not read CSV: {exc}")

    if df.empty:
        raise gr.Error("The uploaded dataset is empty.")

    if len(df.columns) < 1:
        raise gr.Error("No columns were found in the dataset.")

    if len(df) > 200_000:
        raise gr.Error(
            "This version supports up to 200,000 rows per upload. "
            "Please upload a smaller sample."
        )

    # Limit columns to prevent extremely wide uploads from
    # overwhelming the mobile interface.
    if len(df.columns) > 100:
        raise gr.Error(
            "This version supports up to 100 columns. "
            "Please select a smaller subset of columns."
        )

    # Normalize column names only for internal detection.
    # Original column names remain unchanged.
    original_shape = df.shape
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    datetime_columns = []

    # Detect likely date columns.
    for col in df.columns:
        name = str(col).lower()

        if any(word in name for word in [
            "date", "datetime", "timestamp"
        ]):
            try:
                parsed = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

                if parsed.notna().mean() >= 0.60:
                    df[col] = parsed
                    datetime_columns.append(col)

            except Exception:
                pass

    # Exclude dates that pandas also classified as numeric,
    # should such columns exist.
    numeric_columns = [
        col for col in numeric_columns
        if col not in datetime_columns
    ]

    categorical_columns = [
        col for col in df.columns
        if col not in numeric_columns
        and col not in datetime_columns
    ]

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    quality_rows = []

    for col in df.columns:
        quality_rows.append({
            "Column": str(col),
            "Data Type": str(df[col].dtype),
            "Missing Values": int(df[col].isna().sum()),
            "Missing %": round(
                df[col].isna().mean() * 100, 2
            ),
            "Unique Values": int(df[col].nunique(dropna=True))
        })

    quality_df = pd.DataFrame(quality_rows)

    # --------------------------------------------------------
    # AUTOMATIC SUMMARY
    # --------------------------------------------------------

    summary = f"""
# InsightML — Dataset Analysis

## Dataset overview

- **Rows:** {original_shape[0]:,}
- **Columns:** {original_shape[1]:,}
- **Duplicate rows:** {duplicate_rows:,}
- **Missing cells:** {missing_cells:,}
- **Numeric columns:** {len(numeric_columns)}
- **Categorical columns:** {len(categorical_columns)}
- **Detected date columns:** {len(datetime_columns)}

## Data quality

This analysis describes the uploaded dataset. Missing values and
duplicate rows are reported but are not silently removed.

## Available analyses

"""

    if numeric_columns:
        summary += "- Numeric summary statistics\n"
        summary += "- Numeric distribution charts\n"

    if categorical_columns:
        summary += "- Categorical value counts\n"

    if len(numeric_columns) >= 2:
        summary += "- Numeric correlation analysis\n"

    if datetime_columns:
        summary += "- Time-based analysis for detected date columns\n"

    # --------------------------------------------------------
    # NUMERIC SUMMARY
    # --------------------------------------------------------

    if numeric_columns:
        numeric_summary = (
            df[numeric_columns]
            .describe()
            .T.reset_index()
            .rename(columns={"index": "Column"})
            .round(3)
        )
    else:
        numeric_summary = pd.DataFrame({
            "Message": [
                "No numeric columns detected in this dataset."
            ]
        })

    # --------------------------------------------------------
    # NUMERIC DISTRIBUTION CHART
    # --------------------------------------------------------

    if numeric_columns:
        selected_num = numeric_columns[0]

        chart_df = df[[selected_num]].copy()
        chart_df[selected_num] = pd.to_numeric(
            chart_df[selected_num],
            errors="coerce"
        )
        chart_df = chart_df.dropna()

        if not chart_df.empty:
            distribution_chart = px.histogram(
                chart_df,
                x=selected_num,
                title=f"Distribution of {selected_num}",
                marginal="box"
            )
        else:
            distribution_chart = px.scatter(
                title="No valid numeric observations to plot"
            )
    else:
        distribution_chart = px.scatter(
            title="No numeric columns available"
        )

    # --------------------------------------------------------
    # CORRELATION ANALYSIS
    # --------------------------------------------------------

    if len(numeric_columns) >= 2:
        corr = df[numeric_columns].corr(
            numeric_only=True
        )

        correlation_chart = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Numeric Feature Correlations",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1
        )

        correlation_df = corr.round(3).reset_index()
        correlation_df = correlation_df.rename(
            columns={"index": "Column"}
        )
    else:
        correlation_chart = px.scatter(
            title="At least two numeric columns are required"
        )

        correlation_df = pd.DataFrame({
            "Message": [
                "Correlation analysis requires two numeric columns."
            ]
        })

    # --------------------------------------------------------
    # CATEGORICAL ANALYSIS
    # --------------------------------------------------------

    if categorical_columns:
        selected_cat = categorical_columns[0]

        counts = (
            df[selected_cat]
            .fillna("(Missing)")
            .astype(str)
            .value_counts()
            .head(15)
            .rename_axis("Category")
            .reset_index(name="Count")
        )

        category_chart = px.bar(
            counts,
            x="Category",
            y="Count",
            title=f"Top Values: {selected_cat}"
        )

        category_summary = counts
    else:
        category_chart = px.scatter(
            title="No categorical columns available"
        )

        category_summary = pd.DataFrame({
            "Message": [
                "No categorical columns were detected."
            ]
        })

    # --------------------------------------------------------
    # DATE ANALYSIS
    # --------------------------------------------------------

    if datetime_columns:
        date_col = datetime_columns[0]

        date_series = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

        date_df = pd.DataFrame({
            "Date": date_series.dt.date
        }).dropna()

        date_counts = (
            date_df.groupby("Date")
            .size()
            .reset_index(name="Records")
        )

        date_chart = px.line(
            date_counts,
            x="Date",
            y="Records",
            title=f"Records Over Time: {date_col}"
        )

        date_summary = date_counts
    else:
        date_chart = px.scatter(
            title="No suitable date column detected"
        )

        date_summary = pd.DataFrame({
            "Message": [
                "Add a recognizable date column for time-based analysis."
            ]
        })

    # --------------------------------------------------------
    # CONDITIONAL BUSINESS KPI DETECTION
    # --------------------------------------------------------

    lower_map = {
        str(col).lower().strip(): col
        for col in df.columns
    }

    revenue_candidates = [
        "revenue", "sales", "amount", "total_amount",
        "total_sales", "total_revenue"
    ]

    quantity_candidates = [
        "quantity", "units_sold", "units", "qty"
    ]

    revenue_col = next(
        (lower_map[name] for name in revenue_candidates
         if name in lower_map),
        None
    )

    quantity_col = next(
        (lower_map[name] for name in quantity_candidates
         if name in lower_map),
        None
    )

    if revenue_col is not None:
        revenue_values = pd.to_numeric(
            df[revenue_col],
            errors="coerce"
        )

        kpis = [{
            "Metric": f"Total {revenue_col}",
            "Value": round(
                float(revenue_values.sum()), 2
            )
        }, {
            "Metric": f"Average {revenue_col}",
            "Value": round(
                float(revenue_values.mean()), 2
            )
        }]

        if quantity_col is not None:
            quantity_values = pd.to_numeric(
                df[quantity_col],
                errors="coerce"
            )

            kpis.append({
                "Metric": f"Total {quantity_col}",
                "Value": round(
                    float(quantity_values.sum()), 2
                )
            })

        kpi_df = pd.DataFrame(kpis)
    else:
        kpi_df = pd.DataFrame({
            "Message": [
                "No recognized revenue/sales column detected. "
                "Business KPIs were not calculated."
            ]
        })

    # --------------------------------------------------------
    # TARGET COLUMN CHECK
    # --------------------------------------------------------

    target_message = ""

    if target_column and target_column in df.columns:
        target_message = (
            f"\n\n## Selected target column\n\n"
            f"`{target_column}` is selected. "
            "This version does not train a predictive model "
            "from the selected target yet."
        )

    summary += target_message

    # --------------------------------------------------------
    # SAVE DOWNLOADABLE REPORTS
    # --------------------------------------------------------

    safe_name = "uploaded_dataset"

    quality_path = REPORT_DIR / f"{safe_name}_data_quality.csv"
    numeric_path = REPORT_DIR / f"{safe_name}_numeric_summary.csv"
    category_path = REPORT_DIR / f"{safe_name}_category_summary.csv"
    kpi_path = REPORT_DIR / f"{safe_name}_business_kpis.csv"

    quality_df.to_csv(quality_path, index=False)
    numeric_summary.to_csv(numeric_path, index=False)
    category_summary.to_csv(category_path, index=False)
    kpi_df.to_csv(kpi_path, index=False)

    # Create one combined text report for download.
    report_path = REPORT_DIR / f"{safe_name}_analysis_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary)
        f.write("\n\nDATA QUALITY\n")
        f.write(quality_df.to_string(index=False))
        f.write("\n\nNUMERIC SUMMARY\n")
        f.write(numeric_summary.to_string(index=False))
        f.write("\n\nBUSINESS KPIs\n")
        f.write(kpi_df.to_string(index=False))

    # Return all dashboard outputs.
    return (
        summary,
        quality_df,
        numeric_summary,
        distribution_chart,
        correlation_chart,
        correlation_df,
        category_chart,
        category_summary,
        date_chart,
        date_summary,
        kpi_df,
        str(report_path)
    )


# ------------------------------------------------------------
# 3. BUILD THE INTERACTIVE USER INTERFACE
# ------------------------------------------------------------

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
#main-title {
    text-align: center;
}
button {
    transition: all 0.2s ease-in-out !important;
}
button:hover {
    transform: translateY(-2px);
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css,
    title="InsightML | Adaptive Analytics"
) as demo:

    gr.Markdown(
        """
        # 🔍 InsightML
        ### Adaptive Machine Learning & Business Analytics

        Upload a CSV dataset. InsightML inspects the actual columns
        and displays analyses supported by the available data.
        """,
        elem_id="main-title"
    )

    gr.Markdown(
        """
        **Current version:** Adaptive data analysis, quality checks,
        summaries, charts, and conditional sales KPIs.

        Predictive models and advanced ML modules will be integrated
        in a later version. No prediction is generated in this version.
        """
    )

    with gr.Row():
        file_input = gr.File(
            label="Upload your CSV dataset",
            file_types=[".csv"],
            type="filepath"
        )

    target_dropdown = gr.Dropdown(
        label="Optional target column",
        choices=[],
        value=None,
        interactive=True,
        info="For now, this selection is informational only."
    )

    # Populate available target choices from the uploaded file.
    def get_columns(file_path):
        if file_path is None:
            return gr.update(choices=[], value=None)

        try:
            columns = pd.read_csv(
                file_path,
                nrows=5
            ).columns.tolist()

            return gr.update(
                choices=columns,
                value=None
            )
        except Exception:
            return gr.update(choices=[], value=None)

    file_input.change(
        fn=get_columns,
        inputs=file_input,
        outputs=target_dropdown
    )

    run_button = gr.Button(
        "🚀 Analyze Dataset",
        variant="primary"
    )

    summary_output = gr.Markdown(
        label="Dataset Overview"
    )

    with gr.Tabs():

        with gr.Tab("Data Quality"):
            quality_output = gr.Dataframe(
                label="Column Quality Report",
                interactive=False
            )

        with gr.Tab("Numeric Analysis"):
            numeric_output = gr.Dataframe(
                label="Summary Statistics",
                interactive=False
            )
            distribution_output = gr.Plot(
                label="Numeric Distribution"
            )

        with gr.Tab("Correlations"):
            correlation_plot_output = gr.Plot(
                label="Correlation Heatmap"
            )
            correlation_table_output = gr.Dataframe(
                label="Correlation Table",
                interactive=False
            )

        with gr.Tab("Categories"):
            category_plot_output = gr.Plot(
                label="Category Chart"
            )
            category_table_output = gr.Dataframe(
                label="Category Summary",
                interactive=False
            )

        with gr.Tab("Time Analysis"):
            date_plot_output = gr.Plot(
                label="Records Over Time"
            )
            date_table_output = gr.Dataframe(
                label="Date Summary",
                interactive=False
            )

        with gr.Tab("Business KPIs"):
            kpi_output = gr.Dataframe(
                label="Conditional Business KPIs",
                interactive=False
            )

    report_output = gr.File(
        label="Download Analysis Report"
    )

    run_button.click(
        fn=analyze_dataset,
        inputs=[
            file_input,
            target_dropdown
        ],
        outputs=[
            summary_output,
            quality_output,
            numeric_output,
            distribution_output,
            correlation_plot_output,
            correlation_table_output,
            category_plot_output,
            category_table_output,
            date_plot_output,
            date_table_output,
            kpi_output,
            report_output
        ]
    )

    gr.Markdown(
        """
        ---
        **InsightML | Portfolio Project**

        Note: Statistical patterns and detected anomalies are not
        proof of causation, fraud, or future performance. Review
        data quality before using results for business decisions.
        """
    )

# ------------------------------------------------------------
# 4. LAUNCH APPLICATION
# ------------------------------------------------------------

print("\nLaunching InsightML dashboard...")
print("Keep this Colab cell running while using the application.")

demo.launch(
    server_name="0.0.0.0",
    server_port=int(__import__("os").environ.get("PORT", "7860")),
    share=False,
    show_error=True
)
