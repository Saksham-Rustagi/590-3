"""Computation helpers for the California housing dashboard."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _min_max(series: pd.Series) -> pd.Series:
    """Safe min-max scaling; returns zeros if the range is 0 or NaN."""
    s_min, s_max = series.min(), series.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_min == s_max:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)


def monthly_mortgage(principal: pd.Series, annual_rate: float, years: int) -> pd.Series:
    """Compute monthly mortgage payment using standard amortization formula."""
    monthly_rate = annual_rate / 12
    n_payments = years * 12
    if monthly_rate == 0:
        return principal / n_payments
    factor = (1 + monthly_rate) ** n_payments
    return principal * (monthly_rate * factor) / (factor - 1)


def compute_affordability(
    housing_df: pd.DataFrame,
    user_income: float,
    housing_share: float,
    down_payment: float,
    rate: float,
    years: int,
) -> pd.DataFrame:
    """Add affordability calculations to the housing dataframe."""
    df = housing_df.copy()

    principal = df["median_house_value"].clip(lower=0) * (1 - down_payment)
    df["estimated_mortgage"] = monthly_mortgage(principal, rate, years)
    user_monthly_income = max(user_income / 12, 1)  # avoid division by zero

    df["user_budget"] = user_monthly_income * housing_share
    df["affordability_ratio"] = df["estimated_mortgage"] / user_monthly_income
    df["user_affordable"] = df["estimated_mortgage"] <= df["user_budget"]

    # Human-friendly labels for map coloring, dynamically based on housing_share.
    threshold_pct = int(housing_share * 100)
    stretch_threshold = housing_share + 0.1
    stretch_pct = int(stretch_threshold * 100)
    df["affordability_category"] = pd.cut(
        df["affordability_ratio"],
        bins=[-np.inf, housing_share, stretch_threshold, np.inf],
        labels=[
            f"Affordable (<={threshold_pct}%)",
            f"Stretch ({threshold_pct}-{stretch_pct}%)",
            f"Unaffordable (>{stretch_pct}%)",
        ],
    )
    return df


def filter_housing(
    housing_df: pd.DataFrame,
    max_price: float,
    ocean_filters: Optional[Iterable[str]] = None,
    sample_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Filter housing data by price, ocean proximity, and optional sampling."""
    df = housing_df[housing_df["median_house_value"] <= max_price].copy()
    if ocean_filters:
        df = df[df["ocean_proximity"].isin(ocean_filters)]
    if sample_limit and len(df) > sample_limit:
        df = df.sample(sample_limit, random_state=0)
    return df


def compute_county_scores(
    acs_df: pd.DataFrame,
    crime_df: pd.DataFrame,
    w_income: float,
    w_safety: float,
    w_transit: float,
    w_poverty: float,
    w_unemp: float,
    recent_year_start: int = 2010,
) -> pd.DataFrame:
    """Create county desirability scores combining income, safety, and ACS factors."""
    # Normalize all weights so their sum is 1.0 (avoid division-by-zero).
    weight_sum = max(w_income + w_safety + w_transit + w_poverty + w_unemp, 1e-9)
    wi = w_income / weight_sum
    ws = w_safety / weight_sum
    wt = w_transit / weight_sum
    wp = w_poverty / weight_sum
    wu = w_unemp / weight_sum

    crime_filtered = crime_df[
        (crime_df["geotype"].str.upper() == "CO")
        & (crime_df["strata_level_name"].str.lower() == "violent crime total")
        & (crime_df["reportyear"] >= recent_year_start)
    ].copy()

    crime_recent = (
        crime_filtered.groupby("county_key")["rate"]
        .mean()
        .reset_index(name="crime_rate")
    )

    # Align with notebook: keep only counties with crime data (inner join).
    county_scores = acs_df.merge(crime_recent, on="county_key", how="inner")
    if county_scores.empty:
        # Return empty but with expected columns
        for col in ["crime_rate", "income_scaled", "crime_scaled_inv", "desirability_score"]:
            county_scores[col] = pd.Series(dtype=float)
        return county_scores

    county_scores["income_scaled"] = _min_max(county_scores["Income"])
    # Affordability proxy: lower-income counties are treated as more affordable.
    county_scores["afford_scaled"] = 1 - county_scores["income_scaled"]
    crime_scaled = _min_max(county_scores["crime_rate"])
    county_scores["crime_scaled_inv"] = 1 - crime_scaled

    # Additional desirability components from ACS:
    # - Lower poverty and unemployment are better
    # - Higher public transit usage is better
    county_scores["poverty_scaled"] = 1 - _min_max(county_scores["Poverty"])
    county_scores["unemp_scaled"] = 1 - _min_max(county_scores["Unemployment"])
    county_scores["transit_scaled"] = _min_max(county_scores["Transit"])

    county_scores["desirability_score"] = (
        wi * county_scores["afford_scaled"]
        + ws * county_scores["crime_scaled_inv"]
        + wt * county_scores["transit_scaled"]
        + wp * county_scores["poverty_scaled"]
        + wu * county_scores["unemp_scaled"]
    )
    return county_scores


def county_selection_options(county_scores: pd.DataFrame) -> list[str]:
    """Return sorted county names for UI controls."""
    return sorted(county_scores["County"].unique())


def _safe_median(series: pd.Series) -> float | None:
    """Return median or None if the series is empty/NaN."""
    if series.empty:
        return None
    val = pd.to_numeric(series, errors="coerce").median()
    return None if pd.isna(val) else float(val)


def format_currency(value: float | None) -> str:
    """Human-friendly currency formatting."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def format_pct(value: float | None, digits: int = 1) -> str:
    """Human-friendly percentage formatting."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}%"


def get_county_summary(
    county: str,
    acs_county_df: pd.DataFrame,
    housing_df: pd.DataFrame,
) -> dict:
    """Collect common county-level metrics for display."""
    summary: dict[str, float | str | None] = {
        "county": county,
        "county_key": None,
        "median_home_value": None,
        "median_income": None,
        "price_to_income": None,
        "state_median_home_value": None,
        "state_median_income": None,
        "state_price_to_income": None,
        "state_mean_commute": None,
        "poverty": None,
        "unemployment": None,
        "mean_commute": None,
        "transit": None,
        "population": None,
        "acs_row": None,
        "commute_modes": None,
    }

    # Grab ACS county row if present.
    acs_row_df = acs_county_df[acs_county_df["County"] == county]
    if acs_row_df.empty:
        return summary
    acs_row = acs_row_df.iloc[0]
    summary["acs_row"] = acs_row
    county_key = acs_row.get("county_key")
    summary["county_key"] = county_key

    # Core metrics from ACS.
    summary["median_income"] = _safe_median(pd.Series([acs_row.get("Income", None)]))
    summary["poverty"] = _safe_median(pd.Series([acs_row.get("Poverty", None)]))
    summary["unemployment"] = _safe_median(pd.Series([acs_row.get("Unemployment", None)]))
    summary["mean_commute"] = _safe_median(pd.Series([acs_row.get("MeanCommute", None)]))
    summary["transit"] = _safe_median(pd.Series([acs_row.get("Transit", None)]))
    summary["population"] = _safe_median(pd.Series([acs_row.get("TotalPop", None)]))

    # State baselines for comparison.
    summary["state_median_income"] = _safe_median(acs_county_df["Income"])
    summary["state_mean_commute"] = _safe_median(acs_county_df["MeanCommute"]) if "MeanCommute" in acs_county_df.columns else None
    summary["state_price_to_income"] = None  # set later once state home value is known

    # Housing-derived medians.
    if "median_house_value" in housing_df.columns:
        summary["state_median_home_value"] = _safe_median(housing_df["median_house_value"])
        if county_key and "county_key" in housing_df.columns:
            county_housing = housing_df[housing_df["county_key"] == county_key]
            summary["median_home_value"] = _safe_median(county_housing["median_house_value"])

    # Price-to-income ratios.
    if summary["median_home_value"] and summary["median_income"]:
        summary["price_to_income"] = summary["median_home_value"] / summary["median_income"]
    if summary["state_median_home_value"] and summary["state_median_income"]:
        summary["state_price_to_income"] = summary["state_median_home_value"] / summary["state_median_income"]

    # Commute modes for charts (values are percentages).
    mode_cols = {
        "Drive": "Drive alone",
        "Carpool": "Carpool",
        "Transit": "Transit",
        "Walk": "Walk",
        "OtherTransp": "Other transport",
        "WorkAtHome": "Work from home",
    }
    commute_records = []
    for col, label in mode_cols.items():
        if col in acs_row and not pd.isna(acs_row[col]):
            commute_records.append({"mode": label, "value": float(acs_row[col])})
    summary["commute_modes"] = pd.DataFrame(commute_records) if commute_records else None

    return summary

