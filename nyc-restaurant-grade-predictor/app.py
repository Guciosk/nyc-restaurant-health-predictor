import streamlit as st
import pandas as pd
import pydeck as pdk

from src.data_loader import get_data, refresh_data
from src.predictor import predict_restaurant_grade
from src.utils import (
    get_grade_color,
    format_probabilities,
    row_to_model_input,
    normalize_text,
    display_value,
    prepare_map_dataframe,
)


# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="CleanKitchen NYC",
    page_icon="🍽️",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS for clean light theme
# -------------------------------------------------
st.markdown("""
<style>
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Headings */
    h1, h2, h3 {
        font-weight: 600 !important;
    }

    /* Button styling */
    .stButton > button {
        background: #E87C4F;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #D4703F;
    }

    /* Form elements */
    .stSelectbox > div > div {
        border-radius: 6px;
    }

    /* DataFrame */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Grade badge */
    .grade-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 64px;
        height: 64px;
        border-radius: 12px;
        font-size: 32px;
        font-weight: 700;
        color: white;
    }

    .grade-A { background: #7DB87D; }
    .grade-B { background: #E8C84A; }
    .grade-C { background: #8B3A3A; }
    .grade-P { background: #9BA8C4; }
    .grade-Z { background: #D4956A; }

    /* Card container */
    .info-card {
        background: #F8F9FA;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #E9ECEF;
        margin-bottom: 1rem;
    }

    /* Results counter */
    .results-counter {
        background: #F8F9FA;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-top: 1rem;
        border-left: 3px solid #E87C4F;
    }

    .results-counter p {
        margin: 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.title("CleanKitchen NYC")
st.markdown(
    "Explore NYC restaurant inspections and get **AI-powered grade predictions** "
    "based on real inspection data."
)


# -------------------------------------------------
# Load data
# -------------------------------------------------
@st.cache_data
def load_app_data():
    df = get_data()

    # Normalize some text fields for filters
    df["borough"] = df["borough"].astype(str).str.strip().str.title()
    df["cuisine_description"] = df["cuisine_description"].astype(str).str.strip().str.title()

    # Keep only the most recent inspection per restaurant (by camis ID)
    if "camis" in df.columns and "inspection_date" in df.columns:
        df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
        df = df.sort_values("inspection_date", ascending=False)
        df = df.drop_duplicates(subset=["camis"], keep="first")

    return df


df = load_app_data()

if df.empty:
    st.error("No data loaded. Please check your CSV files in the data/ folder.")
    st.stop()


# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("🔎 Filter Restaurants")

# Borough filter
boroughs = ["All"] + sorted(df["borough"].dropna().unique().tolist())
borough_choice = st.sidebar.selectbox("Borough", boroughs, index=0)

# ZIP filter (depends on borough choice)
if borough_choice != "All":
    zip_candidates = df.loc[df["borough"] == borough_choice, "zipcode"].unique()
else:
    zip_candidates = df["zipcode"].unique()

zips = ["All"] + sorted([z for z in zip_candidates if pd.notna(z)])
zip_choice = st.sidebar.selectbox("ZIP code", zips, index=0)

# Cuisine filter
cuisine_list = sorted(df["cuisine_description"].dropna().unique().tolist())
cuisine_choice = st.sidebar.multiselect(
    "Cuisine type",
    options=cuisine_list,
    default=[]
)

# Apply filters
df_filtered = df.copy()

if borough_choice != "All":
    df_filtered = df_filtered[df_filtered["borough"] == borough_choice]

if zip_choice != "All":
    df_filtered = df_filtered[df_filtered["zipcode"] == zip_choice]

if cuisine_choice:
    df_filtered = df_filtered[df_filtered["cuisine_description"].isin(cuisine_choice)]

st.sidebar.markdown(f"""
<div class="results-counter">
    <p>{len(df_filtered):,} restaurants found</p>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Data Management (sidebar)
# -------------------------------------------------
st.sidebar.divider()
st.sidebar.caption("Data Management")
if st.sidebar.button("🔄 Fetch Fresh Data"):
    with st.spinner("Fetching latest data from NYC Open Data API... This may take 1-2 minutes."):
        success, message, count = refresh_data()
        if success:
            st.sidebar.success(f"{message} ({count:,} records)")
            st.cache_data.clear()
            st.rerun()
        else:
            st.sidebar.error(message)


# -------------------------------------------------
# MAIN LAYOUT: Map (left) + Details/Prediction (right)
# -------------------------------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Map of Restaurants")

    if len(df_filtered) == 0:
        st.info("No restaurants match your filters. Try changing the filters.")
    else:
        # Prepare data for PyDeck (no marker limit needed)
        map_df = prepare_map_dataframe(df_filtered)

        center_lat = map_df["latitude"].mean()
        center_lon = map_df["longitude"].mean()

        # Adaptive zoom based on data spread
        lat_range = map_df["latitude"].max() - map_df["latitude"].min()
        lon_range = map_df["longitude"].max() - map_df["longitude"].min()
        max_range = max(lat_range, lon_range)
        zoom = 15 if max_range < 0.01 else 13 if max_range < 0.05 else 12 if max_range < 0.1 else 11

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["longitude", "latitude"],
            get_color="color",
            get_radius=8,
            radius_min_pixels=2,
            radius_max_pixels=6,
            radius_scale=1,
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom,
            pitch=0,
        )

        tooltip = {
            "html": "<b>{name}</b><br/>Cuisine: {cuisine_description}<br/>Borough: {borough}<br/>ZIP: {zipcode}<br/>Score: {score_display}<br/>Grade: <b>{grade_display}</b>",
            "style": {"backgroundColor": "#FFFFFF", "color": "#2C3E50", "fontSize": "14px", "padding": "12px", "borderRadius": "6px", "boxShadow": "0 2px 8px rgba(0,0,0,0.15)"}
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
            ),
            height=500,
            use_container_width=True,
        )

        # Grade legend
        st.markdown("""
        <div style="display: flex; gap: 20px; font-size: 16px; margin-top: 10px; flex-wrap: wrap;">
            <span><span style="color: #7DB87D; font-size: 20px;">●</span> A</span>
            <span><span style="color: #E8C84A; font-size: 20px;">●</span> B</span>
            <span><span style="color: #8B3A3A; font-size: 20px;">●</span> C</span>
            <span><span style="color: #9BA8C4; font-size: 20px;">●</span> Pending</span>
            <span><span style="color: #D4956A; font-size: 20px;">●</span> N/A</span>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"Showing all {len(map_df):,} restaurants on map.")

    st.subheader("Restaurant List")
    st.caption("Filtered view based on your selections in the sidebar.")

    # Show a simpler table
    cols_to_show = [
        c for c in ["DBA", "dba", "borough", "zipcode",
                    "cuisine_description", "score", "grade", "inspection_date"]
        if c in df_filtered.columns
    ]
    if cols_to_show:
        st.dataframe(
            df_filtered[cols_to_show].head(300),
            use_container_width=True
        )
    else:
        st.dataframe(df_filtered.head(300), use_container_width=True)


with right_col:
    st.subheader(" Inspect & Predict")

    if len(df_filtered) == 0:
        st.info("Use the filters to select at least one restaurant.")
    else:
        # Let user pick a restaurant from a dropdown
        # Use name + zip as label
        if "dba" in df_filtered.columns:
            name_col = "dba"
        elif "DBA" in df_filtered.columns:
            name_col = "DBA"
        else:
            name_col = df_filtered.columns[0]  # fallback

        df_filtered = df_filtered.reset_index(drop=True)
        options = df_filtered.index.tolist()
        labels = [
            f"{df_filtered.loc[i, name_col]} ({df_filtered.loc[i, 'borough']}, {df_filtered.loc[i, 'zipcode']})"
            for i in options
        ]

        selected_idx = st.selectbox(
            "Choose a restaurant to analyze:",
            options=options,
            format_func=lambda i: labels[i]
        )

        selected_row = df_filtered.loc[selected_idx]

        st.markdown("###  Selected Restaurant")
        st.markdown(f"**Name:** {selected_row.get(name_col, 'N/A')}")
        st.markdown(f"**Borough:** {selected_row.get('borough', 'N/A')}")
        st.markdown(f"**ZIP:** {selected_row.get('zipcode', 'N/A')}")
        st.markdown(f"**Cuisine:** {selected_row.get('cuisine_description', 'N/A')}")

        # Show existing inspection info (if present)
        st.markdown("###  Latest Inspection Info")
        st.markdown(f"- **Score:** {display_value(selected_row.get('score'), 'N/A')}")
        st.markdown(f"- **Official Grade:** {display_value(selected_row.get('grade'), 'Unavailable')}")
        if "inspection_date" in selected_row:
            st.markdown(f"- **Inspection Date:** {selected_row.get('inspection_date')}")

        st.markdown("---")
        st.markdown("###  Model Prediction")

        if st.button("Predict Inspection Grade", use_container_width=True):
            with st.spinner("Analyzing restaurant data..."):
                try:
                    # Build model input
                    model_input = row_to_model_input(selected_row)
                    result = predict_restaurant_grade(model_input)

                    predicted_grade = result["grade"]
                    probabilities = result["probabilities"]
                    formatted_probs = format_probabilities(probabilities)

                    color = get_grade_color(predicted_grade)

                    # Prediction result card
                    st.markdown(f"""
                    <div class="info-card" style="text-align: center;">
                        <p style="font-size: 0.85rem; margin-bottom: 0.5rem; color: #6C757D;">
                            PREDICTED GRADE
                        </p>
                        <div class="grade-badge grade-{predicted_grade}" style="margin: 0 auto;">
                            {predicted_grade}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Confidence by Grade")
                    for g, p in formatted_probs:
                        grade_color = get_grade_color(g)
                        st.markdown(f"""
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                <span>Grade {g}</span>
                                <span style="font-weight: 500;">{p:.1f}%</span>
                            </div>
                            <div style="background: #E9ECEF; border-radius: 4px; height: 6px; overflow: hidden;">
                                <div style="background: {grade_color}; width: {p}%; height: 100%; border-radius: 4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error making prediction: {e}")
