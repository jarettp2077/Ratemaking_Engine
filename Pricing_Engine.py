import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Classification Ratemaking Engine", layout="wide")


# Load data

FILE = Path("classification_ratemaking_project_data.csv")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Development columns

incurred_cols = [
    "incurred_loss_12",
    "incurred_loss_24",
    "incurred_loss_36",
    "incurred_loss_48",
    "incurred_loss_60"
]

paid_cols = [
    "paid_loss_12",
    "paid_loss_24",
    "paid_loss_36",
    "paid_loss_48",
    "paid_loss_60"
]


# Class name mapping

CLASS_MAP = {
    "A": "Preferred Urban",
    "B": "Standard Urban",
    "C": "Preferred Suburban",
    "D": "Standard Suburban",
    "E": "Preferred Rural",
    "F": "High Hazard Mixed"
}


# Display formatting helpers

MONEY_COLS = {
    "Ultimate Losses",
    "Ultimate Losses + LAE",
    "Pure Premium",
    "Trended Pure Premium",
    "Class Trended Pure Premium",
    "All-Class Trended Pure Premium",
    "Credibility-Weighted Trended Pure Premium",
    "Fixed Expense per Exposure",
    "Current Premium per Exposure",
    "Indicated Premium per Exposure"
}

FOUR_DEC_COLS = {
    "Tail Factor",
    "Overall Frequency from Data",
    "Credibility Z",
    "Trend Factor",
    "Variable Expense Ratio",
    "Profit and Contingency Ratio",
    "Selected LAE Ratio",
    "Selected Frequency Trend",
    "Selected Severity Trend"
}


def round_for_display_df(df):
    df_display = df.copy()

    for col in df_display.columns:
        if col in MONEY_COLS:
            df_display[col] = df_display[col].round(2)
        elif col in FOUR_DEC_COLS:
            df_display[col] = df_display[col].round(4)
        elif col == "Indicated Rate Change":
            df_display[col] = df_display[col].round(4)

    return df_display


def round_for_display_series(series):
    s = series.copy()

    for idx in s.index:
        if idx in MONEY_COLS:
            s.loc[idx] = round(float(s.loc[idx]), 2)
        elif idx in FOUR_DEC_COLS:
            s.loc[idx] = round(float(s.loc[idx]), 4)
        elif idx == "Indicated Rate Change":
            s.loc[idx] = round(float(s.loc[idx]), 4)

    return s


def round_triangle_for_display(df):
    return df.round(2)


def round_ldf_cdf_for_display(series):
    return series.round(4)


# Helper functions

def get_class_df(df, risk_class):
    return df[df["class_id"] == risk_class].copy()


def calculate_full_credibility_standard(
    z_value=1.96,
    tolerance=0.05
):
    N0 = (z_value / tolerance) ** 2
    return N0


def calculate_credibility(class_claims, N0):
    if N0 <= 0:
        return 1.0
    return min(1.0, np.sqrt(class_claims / N0))


def develop_and_trend_subset(
    df_subset,
    dev_cols,
    year_col="accident_year",
    exposure_col="earned_exposures",
    claim_count_col="reported_claim_count",
    tail_factor=1.00
):
    # Build annual aggregated dataframe first
    annual_df = (
        df_subset.groupby(year_col, as_index=True)
        .agg({
            exposure_col: "sum",
            claim_count_col: "sum",
            **{col: "sum" for col in dev_cols}
        })
        .sort_index()
    )

    # Triangle
    triangle = annual_df[dev_cols].copy().replace(0, np.nan)

    # Arithmetic average LDFs
    cols = list(triangle.columns)
    ldf_dict = {}

    for i in range(len(cols) - 1):
        curr_col = cols[i]
        next_col = cols[i + 1]

        link_ratios = triangle[next_col] / triangle[curr_col]
        link_ratios = link_ratios.replace([np.inf, -np.inf], np.nan).dropna()

        ldf_dict[f"{curr_col}_to_{next_col}"] = link_ratios.mean()

    ldf_series = pd.Series(ldf_dict, name="Arithmetic Avg LDF")

    # CDFs to ultimate
    ldf_values = list(ldf_series.values)

    cdfs = [None] * len(cols)
    running = tail_factor
    cdfs[-1] = running

    for i in range(len(cols) - 2, -1, -1):
        running *= ldf_values[i]
        cdfs[i] = running

    cdf_series = pd.Series(cdfs, index=cols, name="CDF_to_Ultimate")

    # Ultimate losses by accident year
    ult_dict = {}

    for ay, row in triangle.iterrows():
        latest_col = row.last_valid_index()
        latest_val = row[latest_col]
        selected_cdf = cdf_series.loc[latest_col]
        ult_dict[ay] = latest_val * selected_cdf

    ult_series = pd.Series(ult_dict, name="Ultimate Losses").sort_index()

    # Exposures and claims by accident year
    exposure_series = annual_df[exposure_col].rename("Earned Exposures")
    claim_count_series = annual_df[claim_count_col].rename("Reported Claim Count")

    # Selected LAE ratio
    lae_ratio = np.average(
        df_subset["lae_provision_ratio_to_losses"],
        weights=df_subset[exposure_col]
    )

    ult_lae_series = (ult_series * (1 + lae_ratio)).rename("Ultimate Losses + LAE")

    # Pure premium by accident year
    pure_premium_series = (ult_lae_series / exposure_series).rename("Pure Premium")

    # Trend assumptions
    freq_trend = np.average(
        df_subset["selected_annual_frequency_trend"],
        weights=df_subset[exposure_col]
    )

    sev_trend = np.average(
        df_subset["selected_annual_severity_trend"],
        weights=df_subset[exposure_col]
    )

    proposed_start = pd.to_datetime(df_subset["proposed_policy_year_start"].iloc[0])
    proposed_end = pd.to_datetime(df_subset["proposed_policy_year_end"].iloc[0])
    future_midpoint = proposed_start + (proposed_end - proposed_start) / 2

    accident_year_midpoints = pd.to_datetime(
        pure_premium_series.index.astype(str) + "-07-01"
    )

    years_trend = (future_midpoint - accident_year_midpoints).days / 365.25

    trend_factor_values = (
        ((1 + freq_trend) ** years_trend) *
        ((1 + sev_trend) ** years_trend)
    )

    trend_factor_series = pd.Series(
        trend_factor_values,
        index=pure_premium_series.index,
        name="Trend Factor"
    )

    trended_pure_premium_series = (
        pure_premium_series * trend_factor_series
    ).rename("Trended Pure Premium")

    # Selected trended pure premium
    selected_trended_pure_premium = (
        (trended_pure_premium_series * exposure_series).sum()
        / exposure_series.sum()
    )

    # Detail output
    detail_df = pd.DataFrame({
        "Ultimate Losses": ult_series,
        "Ultimate Losses + LAE": ult_lae_series,
        "Earned Exposures": exposure_series,
        "Reported Claim Count": claim_count_series,
        "Pure Premium": pure_premium_series,
        "Trend Factor": trend_factor_series,
        "Trended Pure Premium": trended_pure_premium_series
    })

    return {
        "triangle": triangle,
        "ldfs": ldf_series,
        "cdfs": cdf_series,
        "detail": detail_df,
        "selected_trended_pure_premium": selected_trended_pure_premium,
        "selected_lae_ratio": lae_ratio,
        "selected_freq_trend": freq_trend,
        "selected_sev_trend": sev_trend
    }


def pricing_engine_with_credibility(
    df,
    risk_class,
    dev_cols,
    year_col="accident_year",
    exposure_col="earned_exposures",
    claim_count_col="reported_claim_count",
    tail_factor=1.00,
    z_value=1.96,
    tolerance=0.05
):
    # Stricter full credibility standard
    N0 = calculate_full_credibility_standard(
        z_value=z_value,
        tolerance=tolerance
    )

    # Keep observed overall frequency for display only
    overall_frequency = df[claim_count_col].sum() / df[exposure_col].sum()

    df_class = get_class_df(df, risk_class)

    class_result = develop_and_trend_subset(
        df_subset=df_class,
        dev_cols=dev_cols,
        year_col=year_col,
        exposure_col=exposure_col,
        claim_count_col=claim_count_col,
        tail_factor=tail_factor
    )

    class_trended_pp = class_result["selected_trended_pure_premium"]

    all_class_result = develop_and_trend_subset(
        df_subset=df,
        dev_cols=dev_cols,
        year_col=year_col,
        exposure_col=exposure_col,
        claim_count_col=claim_count_col,
        tail_factor=tail_factor
    )

    all_class_trended_pp = all_class_result["selected_trended_pure_premium"]

    class_claims = df_class[claim_count_col].sum()

    # Current premium per exposure (for comparison only)
    current_premium_per_exposure = np.average(
        df_class["current_avg_premium_per_exposure"],
        weights=df_class[exposure_col]
    )

    Z = calculate_credibility(class_claims=class_claims, N0=N0)

    credibility_weighted_pp = (
        Z * class_trended_pp +
        (1 - Z) * all_class_trended_pp
    )

    variable_expense_ratio = df_class["variable_expense_ratio"].iloc[0]
    fixed_expense = df_class["fixed_expense_per_exposure"].iloc[0]
    profit_ratio = df_class["profit_and_contingency_ratio"].iloc[0]

    indicated_premium = (
        credibility_weighted_pp + fixed_expense
    ) / (1 - variable_expense_ratio - profit_ratio)

    # Indicated rate change
    indicated_rate_change = (
        indicated_premium / current_premium_per_exposure
    ) - 1

    summary = pd.Series({
        "Class ID": risk_class,
        "Class Description": CLASS_MAP.get(risk_class, ""),
        "Tail Factor": tail_factor,
        "Overall Frequency from Data": overall_frequency,
        "Current Premium per Exposure": current_premium_per_exposure,
        "Derived Full Credibility Standard N0": N0,
        "Class Reported Claim Count": class_claims,
        "Credibility Z": Z,
        "Class Trended Pure Premium": class_trended_pp,
        "All-Class Trended Pure Premium": all_class_trended_pp,
        "Credibility-Weighted Trended Pure Premium": credibility_weighted_pp,
        "Fixed Expense per Exposure": fixed_expense,
        "Variable Expense Ratio": variable_expense_ratio,
        "Profit and Contingency Ratio": profit_ratio,
        "Indicated Premium per Exposure": indicated_premium,
        "Indicated Rate Change": indicated_rate_change,
        "Selected LAE Ratio": class_result["selected_lae_ratio"],
        "Selected Frequency Trend": class_result["selected_freq_trend"],
        "Selected Severity Trend": class_result["selected_sev_trend"]
    }, name="Premium Summary")

    return {
        "class_triangle": class_result["triangle"],
        "class_ldfs": class_result["ldfs"],
        "class_cdfs": class_result["cdfs"],
        "class_detail": class_result["detail"],
        "all_class_detail": all_class_result["detail"],
        "summary": summary
    }



# Streamlit app

st.title("Classification Ratemaking Engine")
st.write("Select a risk class and review the premium indication, development factors, and detailed calculations.")

if not FILE.exists():
    st.error("classification_ratemaking_project_data.csv was not found in the same folder as app.py.")
    st.stop()

df = load_data(FILE)

all_classes = sorted(df["class_id"].dropna().unique())

st.sidebar.header("Inputs")

class_display = [f"{c} - {CLASS_MAP.get(c, '')}" for c in all_classes]

selected_display = st.sidebar.selectbox(
    "Select Risk Class",
    class_display
)

selected_class = selected_display.split(" - ")[0]

development_basis = st.sidebar.selectbox("Development Basis", ["Incurred", "Paid"])
tail_factor = st.sidebar.number_input("Tail Factor", min_value=1.00, value=1.00, step=0.01)
z_value = st.sidebar.number_input("Z Value", min_value=0.01, value=1.96, step=0.01)
tolerance = st.sidebar.number_input("Tolerance", min_value=0.001, value=0.05, step=0.005, format="%.3f")

dev_cols = incurred_cols if development_basis == "Incurred" else paid_cols

result = pricing_engine_with_credibility(
    df=df,
    risk_class=selected_class,
    dev_cols=dev_cols,
    tail_factor=tail_factor,
    z_value=z_value,
    tolerance=tolerance
)

summary = result["summary"]
summary_display = round_for_display_series(summary)

st.subheader(
    f"Indicated Premium Summary — Class {selected_class} ({CLASS_MAP[selected_class]})"
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Premium per Exposure",
    f"{summary['Current Premium per Exposure']:.2f}"
)

col2.metric(
    "Indicated Premium per Exposure",
    f"{summary['Indicated Premium per Exposure']:.2f}"
)

col3.metric(
    "Indicated Rate Change",
    f"{summary['Indicated Rate Change']:.2%}"
)

st.caption(
    "The indicated rate change compares the modeled premium indication to the current average premium per exposure for the selected class."
)

st.dataframe(summary_display.to_frame(name="Value"), use_container_width=True)

st.subheader("Class Triangle")
st.dataframe(round_triangle_for_display(result["class_triangle"]), use_container_width=True)

st.subheader("Class LDFs")
st.dataframe(round_ldf_cdf_for_display(result["class_ldfs"]).to_frame(), use_container_width=True)

st.subheader("Class CDFs")
st.dataframe(round_ldf_cdf_for_display(result["class_cdfs"]).to_frame(), use_container_width=True)

st.subheader("Class Detail")
st.dataframe(round_for_display_df(result["class_detail"]), use_container_width=True)

st.subheader("All Classes Summary")
summary_list = []

for cls in all_classes:
    class_result = pricing_engine_with_credibility(
        df=df,
        risk_class=cls,
        dev_cols=dev_cols,
        tail_factor=tail_factor,
        z_value=z_value,
        tolerance=tolerance
    )
    summary_list.append(class_result["summary"])

summary_table = pd.DataFrame(summary_list)
summary_table_display = round_for_display_df(summary_table)

st.dataframe(
    summary_table_display[[
        "Class ID",
        "Class Description",
        "Current Premium per Exposure",
        "Derived Full Credibility Standard N0",
        "Class Reported Claim Count",
        "Credibility Z",
        "Class Trended Pure Premium",
        "All-Class Trended Pure Premium",
        "Credibility-Weighted Trended Pure Premium",
        "Indicated Premium per Exposure",
        "Indicated Rate Change"
    ]],
    use_container_width=True
)


# Calculation breakdown

st.divider()
st.header("Calculation Breakdown")

st.subheader("Credibility Calculation")
st.write("Full credibility standard derived using the stricter claim-count-only approach:")

st.latex(r"N_0 = \left(\frac{z}{k}\right)^2")

st.write(f"Z value = {z_value:.4f}")
st.write(f"Tolerance = {tolerance:.4f}")
st.write(f"Observed frequency from data = {summary['Overall Frequency from Data']:.4f}")
st.write(f"Full credibility standard N0 = {summary['Derived Full Credibility Standard N0']:.2f}")

st.latex(r"Z = \min\left(1,\sqrt{\frac{N}{N_0}}\right)")

st.write(f"Class claim count N = {summary['Class Reported Claim Count']}")
st.write(f"Credibility Z = {summary['Credibility Z']:.4f}")

st.divider()

st.subheader("Pure Premium Calculation")
st.latex(r"\text{Pure Premium} = \frac{\text{Ultimate Losses + LAE}}{\text{Earned Exposures}}")
st.write("Accident-year level calculations are shown below.")

st.dataframe(round_for_display_df(result["class_detail"]), use_container_width=True)

st.divider()

st.subheader("Credibility-Weighted Trended Pure Premium")
st.latex(r"\text{Selected PP} = Z \times PP_{class} + (1-Z) \times PP_{all}")

st.write(f"Class trended pure premium = {summary['Class Trended Pure Premium']:.2f}")
st.write(f"All-class trended pure premium = {summary['All-Class Trended Pure Premium']:.2f}")
st.write(
    f"Credibility-weighted trended pure premium = "
    f"{summary['Credibility-Weighted Trended Pure Premium']:.2f}"
)

st.divider()

st.subheader("Trend Selection")
st.write(f"Selected frequency trend = {summary['Selected Frequency Trend']:.4f}")
st.write(f"Selected severity trend = {summary['Selected Severity Trend']:.4f}")

st.divider()

st.subheader("Current vs Indicated Premium")
st.write(f"Current premium per exposure = {summary['Current Premium per Exposure']:.2f}")
st.write(f"Indicated premium per exposure = {summary['Indicated Premium per Exposure']:.2f}")
st.write(f"Indicated rate change = {summary['Indicated Rate Change']:.2%}")

st.divider()

st.subheader("Final Premium Calculation")
st.latex(r"\text{Premium} = \frac{\text{Pure Premium} + \text{Fixed Expense}}{1 - V - Q}")

st.write(f"Fixed Expense per Exposure = {summary['Fixed Expense per Exposure']:.2f}")
st.write(f"Variable Expense Ratio = {summary['Variable Expense Ratio']:.4f}")
st.write(f"Profit and Contingency Ratio = {summary['Profit and Contingency Ratio']:.4f}")

st.success(f"Indicated Premium per Exposure = {summary['Indicated Premium per Exposure']:.2f}")