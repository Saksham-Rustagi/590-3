"""Data loading utilities with simple caching.

This module keeps all I/O in one place so the Streamlit app can stay lean.
All loaders are cached with st.cache_data to avoid repeated disk reads.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_housing(
    path: str = "housing 3.csv",
    county_geojson_path: str | None = "California_Counties_6472744560795949404.geojson",
) -> pd.DataFrame:
    """Load block-level housing data, add convenience columns, and (optionally) county mapping.

    If ``county_geojson_path`` is provided and GeoPandas is installed, we perform a one-time
    spatial join from latitude/longitude points to California county polygons and attach:

    - ``County``: human-readable county name (aligned with ACS naming where possible)
    - ``CountyFIPS`` (if available in the GeoJSON as ``GEOID``)
    """
    # low_memory=False avoids dtype warnings from mixed-type sniffing.
    df = pd.read_csv(path, low_memory=False)
    # Drop rows that would break affordability calculations or mapping.
    df = df.dropna(subset=["median_house_value", "median_income", "longitude", "latitude"])

    # Convert income to dollars; Kaggle source stores it in tens of thousands.
    df["median_income_usd"] = df["median_income"] * 10_000
    df["median_income_k"] = df["median_income_usd"] / 1_000
    df["median_house_value_k"] = df["median_house_value"] / 1_000

    # Optionally enrich with county information via spatial join.
    if county_geojson_path:
        try:
            import geopandas as gpd  # type: ignore[import]

            gdf_points = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                crs="EPSG:4326",
            )

            counties = gpd.read_file(county_geojson_path)
            # Ensure both layers share the same CRS.
            if counties.crs is not None and counties.crs.to_string() != "EPSG:4326":
                counties = counties.to_crs("EPSG:4326")

            # Try to infer the county name column from common candidates.
            name_col = None
            name_candidates = [
                "NAME",
                "Name",
                "County",
                "COUNTY",
                "COUNTY_NAME",
                "County_Name",
                "CNTY_NAME",
                "COUNTY_NAM",
                "NAMELSAD",
                "NAMELSAD10",
                "NAME10",
            ]
            for candidate in name_candidates:
                if candidate in counties.columns:
                    name_col = candidate
                    break
            # Fallback heuristic: any column whose name contains "name".
            if name_col is None:
                for col in counties.columns:
                    if "name" in col.lower():
                        name_col = col
                        break

            if name_col is None:
                st.warning(
                    "County GeoJSON loaded but no obvious county name column was found. "
                    f"Available columns: {list(counties.columns)}. "
                    "Configure `load_housing` with the correct column if needed."
                )
                return df

            keep_cols = [name_col, "geometry"]
            geoid_col = None
            for candidate in ["GEOID", "GEOID10", "COUNTYFP", "COUNTYFP10"]:
                if candidate in counties.columns:
                    geoid_col = candidate
                    break
            has_geoid = geoid_col is not None
            if has_geoid:
                keep_cols.append(geoid_col)
            counties_slim = counties[keep_cols]

            # Spatial join: assign each point to the containing county polygon.
            try:
                joined = gpd.sjoin(gdf_points, counties_slim, how="left", predicate="within")
            except TypeError:
                # Fallback for older GeoPandas versions that use `op` instead of `predicate`.
                joined = gpd.sjoin(gdf_points, counties_slim, how="left", op="within")

            rename_map: dict[str, str] = {name_col: "County"}
            if has_geoid:
                rename_map[geoid_col] = "CountyFIPS"
            joined = joined.rename(columns=rename_map)

            # Normalize county naming to align with ACS and crime data.
            if "County" in joined.columns:
                joined["County"] = (
                    joined["County"]
                    .astype(str)
                    .str.replace(" County", "", regex=False)
                    .str.strip()
                )
                joined["county_key"] = (
                    joined["County"]
                    .str.lower()
                    .str.replace(" county", "", regex=False)
                    .str.strip()
                )

            # Drop geometry so we return a plain DataFrame.
            joined = joined.drop(columns=["geometry"], errors="ignore")
            df = pd.DataFrame(joined)

        except ImportError:
            # GeoPandas not installed: keep going without county mapping.
            st.warning(
                "GeoPandas is not installed; housing data will not be mapped to counties. "
                "Install `geopandas` to enable county-level mapping on the map and tooltips."
            )
        except Exception as exc:  # pragma: no cover - defensive; should not normally trigger
            st.warning(
                f"Could not attach county mapping to housing data: {exc}. "
                "Continuing without county information."
            )

    return df


@st.cache_data(show_spinner=False)
def load_acs_county(path: str = "acs2017_county_data.csv") -> pd.DataFrame:
    """Load county-level ACS data and keep only California."""
    df = pd.read_csv(path)
    df = df[df["State"] == "California"].copy()
    df["county_key"] = (
        df["County"]
        .astype(str)
        .str.lower()
        .str.replace(" county", "", regex=False)
        .str.strip()
    )
    return df


@st.cache_data(show_spinner=False)
def load_acs_tract(path: str = "acs2017_census_tract_data.csv") -> pd.DataFrame:
    """Load tract-level ACS data and keep only California."""
    df = pd.read_csv(path)
    df = df[df["State"] == "California"].copy()
    return df


@st.cache_data(show_spinner=False)
def load_crime(path: str = "crime.csv") -> pd.DataFrame:
    """Load violent crime data."""
    df = pd.read_csv(path, low_memory=False)
    # Normalize text fields for robust filtering/joins.
    df["county_name"] = df["county_name"].astype(str).str.strip()
    df["county_key"] = (
        df["county_name"]
        .str.lower()
        .str.replace(" county", "", regex=False)
        .str.strip()
    )
    df["geotype"] = df["geotype"].astype(str).str.strip()
    df["strata_level_name"] = df["strata_level_name"].astype(str).str.strip()
    # Normalize key numeric fields for reliable filtering.
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df["race_eth_code"] = pd.to_numeric(df["race_eth_code"], errors="coerce")
    df["reportyear"] = pd.to_numeric(df["reportyear"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience loader for the app entrypoint."""
    housing = load_housing()
    acs_county = load_acs_county()
    acs_tract = load_acs_tract()
    crime = load_crime()
    return housing, acs_county, acs_tract, crime

