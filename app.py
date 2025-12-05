"""California Home Buyer Explorer dashboard (Streamlit + Plotly).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure local modules are importable even if Streamlit changes cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_all_data
from metrics import (
    compute_affordability,
    compute_county_scores,
    county_selection_options,
    filter_housing,
    format_currency,
    format_pct,
    get_county_summary,
)


st.set_page_config(
    page_title="California Home Buyer Explorer",
    layout="wide",
    page_icon="🏠",
)

# --- Data --------------------------------------------------------------------
housing_df, acs_county_df, acs_tract_df, crime_df = load_all_data()


# --- Sidebar controls --------------------------------------------------------
st.sidebar.title("Buyer Profile & Filters")
user_income = st.sidebar.number_input(
    "Annual household income (USD)", min_value=20_000, max_value=500_000, value=60_000, step=5_000
)
housing_share = st.sidebar.slider("Max % of income for housing", min_value=0.10, max_value=0.40, value=0.30, step=0.01)
down_payment = st.sidebar.slider("Down payment fraction", min_value=0.0, max_value=0.5, value=0.20, step=0.05)
interest_rate = st.sidebar.slider("Mortgage interest rate (annual)", min_value=0.02, max_value=0.10, value=0.06, step=0.005)
loan_years = st.sidebar.slider("Loan term (years)", min_value=10, max_value=40, value=30, step=5)

ocean_options = sorted(housing_df["ocean_proximity"].unique())
ocean_filters = st.sidebar.multiselect(
    "Preferred ocean proximity", options=ocean_options, default=ocean_options
)

county_choices = county_selection_options(acs_county_df)
target_counties = st.sidebar.multiselect("Target counties", options=county_choices, default=county_choices)

st.sidebar.markdown("---")
sample_limit = st.sidebar.slider("Max points on map (sampling)", min_value=1000, max_value=20000, value=8000, step=1000)


# --- Derived data ------------------------------------------------------------
housing_afford = compute_affordability(
    housing_df,
    user_income=user_income,
    housing_share=housing_share,
    down_payment=down_payment,
    rate=interest_rate,
    years=loan_years,
)

# Apply ocean filters.
housing_filtered = housing_afford.copy()
if ocean_filters:
    housing_filtered = housing_filtered[housing_filtered["ocean_proximity"].isin(ocean_filters)]

# Apply county filters to housing data if county mapping is available.
if "county_key" in housing_filtered.columns and target_counties:
    target_county_keys = [
        c.lower().replace(" county", "").strip() for c in target_counties
    ]
    housing_filtered = housing_filtered[housing_filtered["county_key"].isin(target_county_keys)]

# Sampling for map to stay performant.
map_df = (
    housing_filtered
    if len(housing_filtered) <= sample_limit
    else housing_filtered.sample(sample_limit, random_state=0)
)

# Scatter view can tolerate a bit more; cap to 2x map sample.
scatter_df = housing_filtered.copy()
scatter_df = scatter_df[scatter_df["median_house_value"] < 500_000]
if len(scatter_df) > sample_limit * 2:
    scatter_df = scatter_df.sample(sample_limit * 2, random_state=42)

# Desirability weights (read from session state; sliders live near the table).
w_income = st.session_state.get("w_income", 0.5)
w_safety = st.session_state.get("w_safety", 0.5)
w_transit = st.session_state.get("w_transit", 0.5)
w_poverty = st.session_state.get("w_poverty", 0.5)
w_unemp = st.session_state.get("w_unemp", 0.5)

county_scores = compute_county_scores(
    acs_df=acs_county_df,
    crime_df=crime_df,
    w_income=w_income,
    w_safety=w_safety,
    w_transit=w_transit,
    w_poverty=w_poverty,
    w_unemp=w_unemp,
)

# Restrict county-related displays to user-selected counties and to counties that
# actually have housing data under the current ocean/income filters.
county_scores_display = county_scores.copy()

# First, honor the explicit county multiselect.
county_scores_display = county_scores_display[county_scores_display["County"].isin(target_counties)]

# Then, if we have housing mapped to counties, drop counties that have no homes
# under the current housing filters (e.g., ocean proximity).
if "county_key" in housing_filtered.columns and "county_key" in county_scores_display.columns:
    active_keys = set(housing_filtered["county_key"].dropna().unique())
    if active_keys:
        county_scores_display = county_scores_display[
            county_scores_display["county_key"].isin(active_keys)
        ]
    # Attach median home value per county for scatter x-axis.
    county_home_value = (
        housing_filtered.groupby("county_key")["median_house_value"]
        .median()
        .reset_index(name="MedianHomeValue")
    )
    county_scores_display = county_scores_display.merge(
        county_home_value, on="county_key", how="left"
    )

county_scores_sorted = county_scores_display.sort_values("desirability_score", ascending=False)
top_county = county_scores_sorted.iloc[0]["County"] if not county_scores_sorted.empty else "N/A"

# Compute dominant ocean proximity per county for bar chart coloring
county_ocean_proximity = {}
if "county_key" in housing_filtered.columns and "ocean_proximity" in housing_filtered.columns:
    for county_key in county_scores_sorted["county_key"].dropna().unique():
        county_housing = housing_filtered[housing_filtered["county_key"] == county_key]
        if not county_housing.empty:
            # Get the most common ocean proximity for this county
            dominant_ocean = county_housing["ocean_proximity"].mode()
            if len(dominant_ocean) > 0:
                county_ocean_proximity[county_key] = dominant_ocean.iloc[0]

# Crime aggregates for reuse (mirrors notebook logic: county-level, 2010+).
crime_co_recent = crime_df[
    (crime_df["geotype"].str.upper() == "CO")
    & (crime_df["strata_level_name"].str.lower() == "violent crime total")
    & (crime_df["reportyear"] >= 2010)
].copy()
crime_avg_by_county = (
    crime_co_recent.groupby(["county_key", "county_name"])["rate"]
    .mean()
    .reset_index(name="crime_rate")
    .sort_values("crime_rate", ascending=False)
)


# --- KPIs --------------------------------------------------------------------
st.title("California Home Buyer Explorer")
st.caption(
    "One-page view of affordability, safety, and demographics for California home buyers. "
    "Use the sidebar to set your profile; the charts update instantly."
)

affordable_count = int(housing_filtered[housing_filtered["user_affordable"]].shape[0])
median_affordable_price = (
    housing_filtered.loc[housing_filtered["user_affordable"], "median_house_value"].median()
    if affordable_count > 0
    else None
)

kpi_cols = st.columns(3)
kpi_cols[0].metric("# Affordable areas", f"{affordable_count:,}")
kpi_cols[1].metric("Top recommended county", top_county)
kpi_cols[2].metric(
    "Median price (affordable set)",
    f"${median_affordable_price:,.0f}" if median_affordable_price else "N/A",
)


# --- Map and recommendations -------------------------------------------------
st.markdown("### Homes Map & County Recommendations")
map_col, table_col = st.columns([2, 1])

with map_col:
    color_mode = st.radio(
        "Map color by",
        options=["Affordability status", "Median home value", "Median income"],
        horizontal=True,
    )

    color_args = {}
    if color_mode == "Affordability status":
        color = "affordability_category"
        # Map affordability categories to intuitive colors:
        # - Affordable -> green
        # - Stretch    -> orange
        # - Unaffordable -> red
        afford_values = housing_filtered["affordability_category"].dropna().unique()
        afford_color_map = {}
        for v in afford_values:
            label = str(v)
            if label.startswith("Affordable"):
                afford_color_map[v] = "#2ca02c"  # green
            elif label.startswith("Stretch"):
                afford_color_map[v] = "#ff7f0e"  # orange
            else:
                afford_color_map[v] = "#d62728"  # red
        color_args["color_discrete_map"] = afford_color_map
    elif color_mode == "Median home value":
        color = "median_house_value"
        color_args["color_continuous_scale"] = px.colors.sequential.Viridis
    else:
        color = "median_income_usd"
        color_args["color_continuous_scale"] = px.colors.sequential.Blues_r  # Reversed for darker colors

    if map_df.empty:
        st.info("No homes match the current filters.")
    else:
        # Create display copy with clean column names for hover
        map_display = map_df.copy()
        rename_map = {
            "median_house_value": "Home Value",
            "median_income_usd": "Income",
            "affordability_ratio": "Affordability Ratio",
            "ocean_proximity": "Ocean Proximity",
            "housing_median_age": "Housing Age",
        }
        map_display = map_display.rename(columns=rename_map)

        hover_data = {
            "Home Value": ":$,",
            "Income": ":$,",
            "Affordability Ratio": ":.0%",
            "Ocean Proximity": True,
            "Housing Age": True,
        }
        # Include county in tooltip if the mapping is available.
        if "County" in map_display.columns:
            hover_data["County"] = True

        # Map color column to renamed version if applicable
        color_col = rename_map.get(color, color)
        
        fig_map = px.scatter_mapbox(
            map_display,
            lat="latitude",
            lon="longitude",
            color=color_col,
            hover_data=hover_data,
            zoom=5,
            height=600,
            size_max=6,
            **color_args,
        )
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=0, b=0),
            legend_title="",
        )
        st.plotly_chart(fig_map, width="stretch")

with table_col:
    st.markdown("**Desirability weights**")
    # Arrange sliders in a 2-column x 3-row grid
    weight_col1, weight_col2 = st.columns(2)
    
    with weight_col1:
        st.slider(
            "Weight: Affordability (lower income)",
            min_value=0.0,
            max_value=1.0,
            value=w_income,
            step=0.05,
            key="w_income",
            help="Higher values prioritize more affordable (typically lower-income) counties.",
        )
        st.slider(
            "Weight: Transit use",
            min_value=0.0,
            max_value=1.0,
            value=w_transit,
            step=0.05,
            key="w_transit",
            help="Higher values prioritize counties where more people commute by transit.",
        )
        st.slider(
            "Weight: Low unemployment",
            min_value=0.0,
            max_value=1.0,
            value=w_unemp,
            step=0.05,
            key="w_unemp",
            help="Higher values prioritize counties with lower unemployment.",
        )
    
    with weight_col2:
        st.slider(
            "Weight: Safety (low crime)",
            min_value=0.0,
            max_value=1.0,
            value=w_safety,
            step=0.05,
            key="w_safety",
            help="Higher values prioritize counties with lower violent crime.",
        )
        st.slider(
            "Weight: Low poverty",
            min_value=0.0,
            max_value=1.0,
            value=w_poverty,
            step=0.05,
            key="w_poverty",
            help="Higher values prioritize counties with lower poverty rates.",
        )

    st.markdown("**County desirability ranking**")
    if county_scores_sorted.empty:
        st.info("No counties selected.")
    else:
        # Prepare data for bar chart with ocean proximity coloring
        # Limit to top 12 counties and keep highest scores at the top of the chart
        rank_df = (
            county_scores_sorted.sort_values("desirability_score", ascending=False)
            .head(12)
            .copy()
        )
        
        # Add dominant ocean proximity column for coloring
        ocean_color_map = {
            "INLAND": "#636363",
            "<1H OCEAN": "#bdd7e7",
            "NEAR OCEAN": "#6baed6",
            "NEAR BAY": "#2171b5",
            "ISLAND": "#ff7f00",
        }
        
        rank_df["dominant_ocean"] = rank_df["county_key"].map(county_ocean_proximity).fillna("INLAND")
        
        fig_rank = px.bar(
            rank_df,
            x="desirability_score",
            y="County",
            orientation="h",
            color="dominant_ocean",
            color_discrete_map=ocean_color_map,
            labels={"desirability_score": "Desirability score", "dominant_ocean": "Ocean proximity"},
        )
        fig_rank.update_layout(
            yaxis_categoryorder="array",
            # Reverse categories using values (avoid Series indexing issues with reversed())
            yaxis_categoryarray=list(rank_df["County"].tolist()[::-1]),
            showlegend=True,
            legend_title="Ocean proximity",
            height=400,
        )
        st.plotly_chart(fig_rank, use_container_width=True)


# --- Relationship & distribution plots --------------------------------------
st.markdown("### Market Relationships")
tab_heatmap, tab_ocean = st.tabs(["Income vs Home Value", "Price by Ocean Proximity"])

with tab_heatmap:
    if scatter_df.empty:
        st.info("No data for current filters.")
    else:
        # Create display copy with clean column names
        scatter_display = scatter_df.copy()
        scatter_display = scatter_display.rename(columns={
            "median_income_k": "Income (k USD)",
            "median_house_value_k": "Home Value (k USD)",
            "ocean_proximity": "Ocean Proximity",
        })

        hover_scatter = {
            "Income (k USD)": ":,.0f",
            "Home Value (k USD)": ":,.0f",
            "Ocean Proximity": True,
        }
        # Include county in tooltip if available.
        if "County" in scatter_display.columns:
            hover_scatter["County"] = True

        # Define a consistent color mapping for ocean proximity:
        # use a continuous-feeling blue gradient for distance-to-water
        # categories and a distinct color for islands.
        ocean_color_map = {
            "INLAND": "#636363",        # grey
            "<1H OCEAN": "#bdd7e7",     # light blue
            "NEAR OCEAN": "#6baed6",    # medium blue
            "NEAR BAY": "#2171b5",      # dark blue
            "ISLAND": "#ff7f00",        # orange highlight
        }

        fig_scatter_income = px.scatter(
            scatter_display,
            x="Income (k USD)",
            y="Home Value (k USD)",
            color="Ocean Proximity",
            color_discrete_map=ocean_color_map,
            opacity=0.35,
            hover_data=hover_scatter,
            labels={
                "Income (k USD)": "Median income (k USD)",
                "Home Value (k USD)": "Median home value (k USD)",
                "Ocean Proximity": "Ocean proximity",
            },
        )
        # Calculate maximum affordable home value based on user budget
        user_monthly_income = user_income / 12
        user_budget = user_monthly_income * housing_share
        monthly_rate = interest_rate / 12
        n_payments = loan_years * 12
        
        if monthly_rate == 0:
            # Simple case: no interest
            max_affordable_value = user_budget * n_payments / (1 - down_payment)
        else:
            # Reverse mortgage formula to find max affordable home value
            factor = (1 + monthly_rate) ** n_payments
            mortgage_multiplier = (monthly_rate * factor) / (factor - 1)
            max_affordable_value = user_budget / ((1 - down_payment) * mortgage_multiplier)
        
        fig_scatter_income.add_hline(
            y=max_affordable_value / 1_000,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Affordability threshold ({int(max_affordable_value/1000)}k)",
            annotation_position="top right",
        )
        fig_scatter_income.add_vline(
            x=user_income / 1_000,
            line_dash="dash",
            line_color="red",
            annotation_text="Your income",
            annotation_position="top right",
        )

        st.plotly_chart(fig_scatter_income, width="stretch")

with tab_ocean:
    if housing_filtered.empty:
        st.info("No data for current filters.")
    else:
        # Create display copy with clean column names
        box_display = housing_filtered.copy()
        box_display = box_display.rename(columns={
            "median_house_value": "Home Value",
            "ocean_proximity": "Ocean Proximity",
        })

        ocean_color_map = {
            "INLAND": "#636363",
            "<1H OCEAN": "#bdd7e7",
            "NEAR OCEAN": "#6baed6",
            "NEAR BAY": "#2171b5",
            "ISLAND": "#ff7f00",
        }

        fig_box = px.box(
            box_display,
            x="Ocean Proximity",
            y="Home Value",
            color="Ocean Proximity",
            color_discrete_map=ocean_color_map,
            labels={"Home Value": "Median home value (USD)", "Ocean Proximity": "Ocean proximity"},
        )
        # Calculate maximum affordable home value based on user budget (same as scatter plot)
        user_monthly_income = user_income / 12
        user_budget = user_monthly_income * housing_share
        monthly_rate = interest_rate / 12
        n_payments = loan_years * 12
        
        if monthly_rate == 0:
            # Simple case: no interest
            max_affordable_value = user_budget * n_payments / (1 - down_payment)
        else:
            # Reverse mortgage formula to find max affordable home value
            factor = (1 + monthly_rate) ** n_payments
            mortgage_multiplier = (monthly_rate * factor) / (factor - 1)
            max_affordable_value = user_budget / ((1 - down_payment) * mortgage_multiplier)
        
        fig_box.add_hline(
            y=max_affordable_value,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Affordability threshold (${int(max_affordable_value/1000)}k)",
            annotation_position="top right",
        )
        st.plotly_chart(fig_box, width="stretch")


# --- Safety & county comparison ---------------------------------------------
st.markdown("### Safety & County Comparison")
# Shared county highlights for all charts in this section.
default_crime_counties = [
    "Los Angeles",
    "San Diego",
    "Orange",
    "Riverside",
    "San Francisco",
]
highlight_counties = st.multiselect(
    "Counties to highlight",
    options=county_choices,
    default=[c for c in default_crime_counties if c in county_choices],
)
# Normalize selection to match crime_df county_key (lowercase, no 'county' suffix).
highlight_keys = [c.lower().replace(" county", "").strip() for c in highlight_counties]

tab_crime_bar, tab_crime_trend, tab_scatter = st.tabs(
    [
        "Crime rate (2010–2013 avg)",
        "Violent crime trend",
        "Income vs poverty",
    ]
)

with tab_crime_bar:
    if crime_avg_by_county.empty:
        st.info("No crime data available.")
    else:
        crime_bar_df = crime_avg_by_county.copy()
        # Show only top 8 and bottom 8 counties by crime rate for readability,
        # but always include any highlighted counties even if they are not in
        # those extremes.
        crime_bar_df["is_highlight"] = crime_bar_df["county_key"].isin(highlight_keys)
        crime_bar_df_sorted = crime_bar_df.sort_values("crime_rate", ascending=False)
        top_8 = crime_bar_df_sorted.head(8)
        bottom_8 = crime_bar_df_sorted.tail(8)
        highlight_rows = crime_bar_df[crime_bar_df["is_highlight"]]
        crime_bar_display = (
            pd.concat([top_8, bottom_8, highlight_rows])
            .drop_duplicates(subset="county_key")
        )

        # Classify bars as top/bottom/highlighted for coloring & legend.
        # If a county is both in top/bottom 8 and highlighted, it should be
        # treated as *highlighted* so it visibly moves into that group.
        top_keys = set(top_8["county_key"])
        bottom_keys = set(bottom_8["county_key"])
        crime_bar_display["group"] = "Highlighted (selected)"
        crime_bar_display.loc[
            (~crime_bar_display["county_key"].isin(highlight_keys))
            & (crime_bar_display["county_key"].isin(top_keys)),
            "group",
        ] = "Top 8 (highest crime)"
        crime_bar_display.loc[
            (~crime_bar_display["county_key"].isin(highlight_keys))
            & (crime_bar_display["county_key"].isin(bottom_keys)),
            "group",
        ] = "Bottom 8 (lowest crime)"

        fig_crime_bar = px.bar(
            crime_bar_display,
            x="crime_rate",
            y="county_name",
            orientation="h",
            labels={"crime_rate": "Violent crimes per 1,000 (avg 2010–2013)", "county_name": "County"},
            color="group",
            color_discrete_map={
                "Top 8 (highest crime)": "#fcae91",       # pastel red
                "Bottom 8 (lowest crime)": "#c7e9c0",    # pastel green
                "Highlighted (selected)": "#fc8d59",     # stronger accent
            },
        )
        fig_crime_bar.update_layout(showlegend=True)
        state_avg = crime_avg_by_county["crime_rate"].mean()
        fig_crime_bar.add_vline(
            x=state_avg,
            line_dash="dash",
            line_color="red",
        )
        st.plotly_chart(fig_crime_bar, width="stretch")

with tab_crime_trend:
    crime_filtered = crime_df[
        (crime_df["geotype"].str.upper() == "CO")
        & (crime_df["strata_level_name"].str.lower() == "violent crime total")
        & (crime_df["county_key"].isin(highlight_keys))
    ]
    if crime_filtered.empty:
        st.info("No crime data for selected counties.")
    else:
        fig_crime = px.line(
            crime_filtered,
            x="reportyear",
            y="rate",
            color="county_name",
            labels={"reportyear": "Year", "rate": "Violent crimes per 1,000"},
        )
        st.plotly_chart(fig_crime, width="stretch")

with tab_scatter:
    scatter_df = county_scores_display.copy()
    if scatter_df.empty:
        st.info("No counties selected.")
    else:
        scatter_df["is_highlight"] = scatter_df["County"].isin(highlight_counties)
        # Use a square root scale so smaller counties remain visible; clip to avoid zeros.
        scatter_df["PopSize"] = np.sqrt(scatter_df["TotalPop"].clip(lower=1))
        fig_scatter = px.scatter(
            scatter_df,
            x="MedianHomeValue",
            y="Poverty",
            size="PopSize",
            size_max=55,
            hover_name="County",
            labels={
                "MedianHomeValue": "Median home value (USD)",
                "Poverty": "Poverty rate (%)",
            },
            color="is_highlight",
            color_discrete_map={True: "#e6550d", False: "#3182bd"},
            text=scatter_df["County"].where(scatter_df["is_highlight"], ""),
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(height=520, showlegend=False)
        st.plotly_chart(fig_scatter, width="stretch")


# --- County detail (buyer view) ------------------------------------------------
st.markdown("### County detail (buyer view)")
selected_county = st.selectbox(
    "Choose a county to inspect",
    options=county_choices,
    index=county_choices.index(top_county) if top_county in county_choices else 0,
)

county_summary = get_county_summary(selected_county, acs_county_df, housing_df)
county_key = county_summary.get("county_key")
tract_df = acs_tract_df[acs_tract_df["County"] == selected_county]
county_housing = (
    housing_df[housing_df["county_key"] == county_key]
    if county_key and "county_key" in housing_df.columns
    else pd.DataFrame()
)

if county_summary.get("acs_row") is None:
    st.info("No county data available for this selection.")
else:
    price_delta = None
    if county_summary["price_to_income"] and county_summary["state_price_to_income"]:
        diff = county_summary["price_to_income"] - county_summary["state_price_to_income"]
        price_delta = f"{diff:+.2f} vs CA"

    commute_delta = None
    if county_summary["mean_commute"] and county_summary["state_mean_commute"]:
        diff = county_summary["mean_commute"] - county_summary["state_mean_commute"]
        commute_delta = f"{diff:+.1f} mins vs CA"

    kpi_cols = st.columns(4)
    kpi_cols[0].metric(
        "Median home value",
        format_currency(county_summary["median_home_value"]),
        delta=(
            f"{county_summary['median_home_value'] - county_summary['state_median_home_value']:+,.0f}"
            if county_summary["median_home_value"] and county_summary["state_median_home_value"]
            else None
        ),
        help="Median of home values for this county vs statewide median.",
    )
    kpi_cols[1].metric(
        "Median household income",
        format_currency(county_summary["median_income"]),
        delta=(
            f"{county_summary['median_income'] - county_summary['state_median_income']:+,.0f}"
            if county_summary["median_income"] and county_summary["state_median_income"]
            else None
        ),
        help="ACS 2017 median household income for the county.",
    )
    kpi_cols[2].metric(
        "Price-to-income ratio",
        f"{county_summary['price_to_income']:.2f}x" if county_summary["price_to_income"] else "N/A",
        delta=price_delta,
        help="Median home value divided by median household income.",
    )
    kpi_cols[3].metric(
        "Mean commute time",
        f"{county_summary['mean_commute']:.1f} mins" if county_summary["mean_commute"] else "N/A",
        delta=commute_delta,
        help="Average one-way commute time for workers in the county.",
    )
    st.markdown(
        f"**Poverty:** {format_pct(county_summary['poverty'])} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Unemployment:** {format_pct(county_summary['unemployment'])} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Transit usage:** {format_pct(county_summary['transit'])}"
    )

    tab_overview, tab_demo, tab_market, tab_qol = st.tabs(
        ["Overview", "Demographics", "Market & Housing", "Quality of Life"]
    )

    with tab_overview:
        st.subheader("Affordability snapshot")
        compare_cols = st.columns(2)
        if county_summary["median_home_value"] and county_summary["state_median_home_value"]:
            compare_df = pd.DataFrame(
                {
                    "Location": ["Selected county", "California median"],
                    "Median home value": [
                        county_summary["median_home_value"],
                        county_summary["state_median_home_value"],
                    ],
                }
            )
            fig_afford = px.bar(
                compare_df,
                x="Location",
                y="Median home value",
                color="Location",
                color_discrete_sequence=["#6baed6", "#9ecae1"],
                labels={"Median home value": "Median home value (USD)"},
            )
            fig_afford.update_layout(showlegend=False)
            compare_cols[0].plotly_chart(fig_afford, use_container_width=True)
        else:
            compare_cols[0].info("Median home value not available.")

        if county_summary["price_to_income"] and county_summary["state_price_to_income"]:
            price_df = pd.DataFrame(
                {
                    "Location": ["Selected county", "California median"],
                    "Price-to-income": [
                        county_summary["price_to_income"],
                        county_summary["state_price_to_income"],
                    ],
                }
            )
            fig_price_ratio = px.bar(
                price_df,
                x="Location",
                y="Price-to-income",
                color="Location",
                color_discrete_sequence=["#3182bd", "#bdd7e7"],
                labels={"Price-to-income": "Price-to-income (x)"},
            )
            fig_price_ratio.update_layout(showlegend=False)
            compare_cols[1].plotly_chart(fig_price_ratio, use_container_width=True)
        else:
            compare_cols[1].info("Price-to-income ratio not available.")

        st.subheader("Income & housing distributions")
        dist_cols = st.columns(2)
        if tract_df.empty:
            dist_cols[0].info("No tract-level income data for this county.")
        else:
            fig_income = px.histogram(
                tract_df,
                x="Income",
                nbins=40,
                color_discrete_sequence=["#3182bd"],
                labels={"Income": "Tract median household income"},
            )
            if county_summary["state_median_income"]:
                fig_income.add_vline(
                    x=county_summary["state_median_income"],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="CA median income",
                    annotation_position="top right",
                )
            dist_cols[0].plotly_chart(fig_income, use_container_width=True)

        if county_housing.empty:
            dist_cols[1].info("No housing value data for this county.")
        else:
            housing_display = county_housing
            if len(housing_display) > 15000:
                housing_display = housing_display.sample(15000, random_state=0)
            fig_price_hist = px.histogram(
                housing_display,
                x="median_house_value",
                nbins=40,
                color_discrete_sequence=["#9ecae1"],
                labels={"median_house_value": "Median home value (USD)"},
            )
            if county_summary["state_median_home_value"]:
                fig_price_hist.add_vline(
                    x=county_summary["state_median_home_value"],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="CA median",
                    annotation_position="top right",
                )
            dist_cols[1].plotly_chart(fig_price_hist, use_container_width=True)

    with tab_demo:
        st.subheader("Population makeup")
        acs_row = county_summary["acs_row"]
        demo_cols = st.columns(2)
        race_cols = ["Hispanic", "White", "Black", "Asian", "Native", "Pacific"]
        race_records = []
        for col in race_cols:
            if col in acs_row and not pd.isna(acs_row[col]):
                race_records.append({"group": col, "value": float(acs_row[col])})
        if race_records:
            race_df = pd.DataFrame(race_records)
            fig_race = px.bar(
                race_df,
                x="group",
                y="value",
                color="group",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"group": "Group", "value": "% of population"},
            )
            fig_race.update_layout(showlegend=False)
            demo_cols[0].plotly_chart(fig_race, use_container_width=True)
        else:
            demo_cols[0].info("Race/ethnicity breakdown not available.")

        if tract_df.empty:
            demo_cols[1].info("No tract-level poverty data for this county.")
        else:
            fig_poverty = px.histogram(
                tract_df,
                x="Poverty",
                nbins=40,
                color_discrete_sequence=["#6baed6"],
                labels={"Poverty": "Tract poverty rate (%)"},
            )
            demo_cols[1].plotly_chart(fig_poverty, use_container_width=True)

    with tab_market:
        st.subheader("Housing market signals")
        market_cols = st.columns(2)
        if county_housing.empty:
            market_cols[0].info("No housing data for this county.")
            market_cols[1].info("No housing data for this county.")
        else:
            if "ocean_proximity" in county_housing.columns:
                fig_price_box = px.box(
                    county_housing,
                    x="ocean_proximity",
                    y="median_house_value",
                    labels={
                        "median_house_value": "Median home value (USD)",
                        "ocean_proximity": "Ocean proximity",
                    },
                    color="ocean_proximity",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                market_cols[0].plotly_chart(fig_price_box, use_container_width=True)
            else:
                market_cols[0].info("Ocean proximity not available for this county.")

            color_arg = "ocean_proximity" if "ocean_proximity" in county_housing.columns else None
            fig_price_scatter = px.scatter(
                county_housing,
                x="median_income_usd",
                y="median_house_value",
                color=color_arg,
                labels={
                    "median_income_usd": "Median household income (USD, block)",
                    "median_house_value": "Median home value (USD)",
                },
                title=None,
                opacity=0.6,
            )
            market_cols[1].plotly_chart(fig_price_scatter, use_container_width=True)

    with tab_qol:
        st.subheader("Commute & transit")
        qol_cols = st.columns(2)
        commute_modes_df = county_summary.get("commute_modes")
        if commute_modes_df is not None and not commute_modes_df.empty:
            fig_commute_modes = px.bar(
                commute_modes_df,
                x="mode",
                y="value",
                color="mode",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                labels={"mode": "Commute mode", "value": "% of workers"},
            )
            fig_commute_modes.update_layout(showlegend=False)
            qol_cols[0].plotly_chart(fig_commute_modes, use_container_width=True)
        else:
            qol_cols[0].info("Commute mode breakdown not available.")

        if tract_df.empty:
            qol_cols[1].info("No tract-level commute data for this county.")
        else:
            fig_commute = px.histogram(
                tract_df,
                x="MeanCommute",
                nbins=30,
                color_discrete_sequence=["#74c476"],
                labels={"MeanCommute": "Mean commute time (minutes)"},
            )
            if county_summary["mean_commute"]:
                fig_commute.add_vline(
                    x=county_summary["mean_commute"],
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="County average",
                    annotation_position="top right",
                )
            if county_summary["state_mean_commute"]:
                fig_commute.add_vline(
                    x=county_summary["state_mean_commute"],
                    line_dash="dot",
                    line_color="red",
                    annotation_text="CA average",
                    annotation_position="top left",
                )
            qol_cols[1].plotly_chart(fig_commute, use_container_width=True)


# --- Footer ------------------------------------------------------------------
st.caption(
    "Data: California housing blocks (Kaggle), ACS 2017 county & tract, "
    "California violent crime (2000–2013). All calculations are cached for responsiveness."
)

