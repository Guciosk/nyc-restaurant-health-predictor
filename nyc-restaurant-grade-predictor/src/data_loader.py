import pandas as pd
import streamlit as st

from .data_pipeline import get_or_fetch_data, refresh_data, has_cached_data, get_cache_info


# -------------------------------------------------
# Load restaurant data with Streamlit caching
# -------------------------------------------------

@st.cache_data
def load_restaurant_data():
    """
    Loads the cleaned restaurant inspection dataset.

    Data is fetched from NYC Open Data API if not cached locally.
    Includes: borough, zipcode, cuisine_description, score,
    critical_flag, and coordinates for mapping.
    """
    # Get data from pipeline (fetches from API if no cache)
    df = get_or_fetch_data()

    # Rename 'boro' to 'borough' for consistency
    if 'boro' in df.columns and 'borough' not in df.columns:
        df = df.rename(columns={'boro': 'borough'})

    # Clean/standardize core fields
    df['borough'] = df['borough'].astype(str).str.strip().str.title()
    df['cuisine_description'] = df['cuisine_description'].astype(str).str.strip().str.title()

    # Clean zipcode - empty strings become None
    df['zipcode'] = df['zipcode'].str.strip().replace('', None)

    # Create critical_flag_bin from critical_flag if not present
    if 'critical_flag_bin' not in df.columns and 'critical_flag' in df.columns:
        df['critical_flag_bin'] = (df['critical_flag'].str.lower() == 'critical').astype(int)

    return df


# -------------------------------------------------
# Public function app will import
# -------------------------------------------------

def get_data():
    """Returns the cleaned restaurant inspection data."""
    return load_restaurant_data()
