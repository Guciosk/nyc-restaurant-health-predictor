import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import datetime

from src.data_loader import get_data, get_raw_data, refresh_data, load_training_data, clear_data_cache
from src.predictor import predict_restaurant_grade, clear_model_cache, get_model_metadata, model_needs_retraining, ModelNeedsRetrainingError
from src.feature_engineering import compute_all_features
from src.trainer import train_model, save_model, get_feature_importance_ranking
from src.utils import (
    get_grade_color,
    format_probabilities,
    row_to_model_input,
    normalize_text,
    display_value,
    prepare_map_dataframe,
)


# --- Initialize Session State for Page Navigation ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def navigate_to(page_name):
    """Function to change the current page in session state."""
    st.session_state.page = page_name

# --- Page Configuration ---
st.set_page_config(
    page_title="CleanKitchen NYC",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    /* 1. FORCE WHITE BACKGROUND */
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    

    /* Remove top padding to align navbar better */
    .block-container {
        padding-top: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* 2. NAVBAR STYLING */
    .brand {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 50px;
        font-weight: 800;
        color: #000;
        text-decoration: none;
        border: none;
        text-border: none !important;
        oncursor: pointer;
    }

    .nav-container {
        display: flex;
        justify-content: center;
        gap: 30px;
        padding-top: 10px;
    }

    /* Change nav-link to be a clickable button-like element */
    .nav-link {
        font-family: sans-serif;
        font-size: 16px;
        font-weight: 600;
        text-transform: uppercase;
        color: #333;
        text-decoration: none;
        letter-spacing: 0.5px;
        cursor: pointer; /* Indicates it's clickable */
        border: none;
        text-border: none !important;
        border-radius: 0px !important;
    }

    /* 3. BUTTON BORDERS (The "Pill" Shape) */
    .stButton > button {
        background-color: transparent !important;
        color: black !important;
        border: 1px solid #000 !important;
        border-radius: 30px !important; /* This creates the rounded 'pill' border */
        padding: 5px 25px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease;
        font-size: 20px !important;
    }
    
    
    .stButton > button:hover {
        background-color: #f0f0f0 !important;
        border-color: #000 !important;
    }
    
    .title-button > button {
        background-color: #007BFF; /* Your desired color */
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 24px;
        font-weight: bold;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
}

    /* 4. HEADER DIVIDER (The horizontal line) */
    .header-separator {
        border-bottom: 1px solid #E0E0E0;
        margin-top: 10px;
        margin-bottom: 40px;
    }

    /* Hero Text */
    .hero-sub {
        font-size: 12px;
        font-weight: 700;
        color: #333;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 15px;
    }
    
    .hero-title {
        font-family: sans-serif;
        font-size: 64px;
        font-weight: 500;
        line-height: 1.1;
        margin-bottom: 40px;
        color: #000;
    }

    /* Image Slots */
    .img-placeholder {
        background-color: #F5F5F5;
        height: 350px; /* Taller to match the screenshot */
        display: flex;
        align-items: center;
        justify-content: center;
        color: #999;
        font-size: 14px;
        border-radius: 2px;
    }
    
    /* ---------------Moving Images --------------- */
    /* The Marquee Effect requires CSS Animation
*/
        .marquee-container {
            overflow: hidden; 
            white-space: nowrap; 
            width: 100%;
            margin: 20px 0;
            border-top: 1px solid #E0E0E0; 
            border-bottom: 1px solid #E0E0E0;
        }

        .marquee-content {
            display: inline-block;
            animation: scroll 50s linear infinite; 
        }

        @keyframes scroll {
            0% {
                transform: translateX(0%);
            }
            100% {
                transform: translateX(-50%);
            }
        }

        /* Updated style for the image items */
        .marquee-item {
            display: inline-block;
            height: 400px; /* Define height for the photo strip */
            width: 600px; /* Define width for each photo */
            margin-right: 30px; /* Space between photos */
            background-color: white; /* Remove placeholder background */
            border-radius: 30px;
            overflow: hidden; 
            vertical-align: top;
        }

    
        .marquee-item img {
            width: 100%;
            height: 100%;
            object-fit: cover; /* Ensures the image covers the area without distortion */
            border-radius: 35px;
        }
.marquee-wrapper {
    position: relative;
    width: 100%;
    mask-image: linear-gradient(to right, 
        transparent, 
        black 10%, 
        black 90%, 
        transparent
    );
    -webkit-mask-image: linear-gradient(to right, 
        transparent, 
        black 10%, 
        black 90%, 
        transparent
    );
}
    
    </style>
""", unsafe_allow_html=True)

#----------------------Load Data---------------
@st.cache_data
def load_app_data():
    """Load feature-enriched restaurant data (one row per restaurant)."""
    # NOTE: 'get_data()' is imported from 'src.data_loader'
    df = get_data() 

    # Normalize text fields for filters 
    df["borough"] = df["borough"].astype(str).str.strip().str.title()
    df["cuisine_description"] = df["cuisine_description"].astype(str).str.strip().str.title()

    # Ensure inspection_date is datetime
    if "inspection_date" in df.columns:
        df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")

    return df


df = load_app_data()

if df.empty:
    st.error("No data loaded. Please check your CSV files in the data/ folder.")
    st.stop()

# --- Navigation Section (Always rendered) ---
col1, col2, col3 = st.columns([2, 5, 1])

with col1:
    # Use a dummy button styled as text for the brand/home link
    st.markdown('<div class="brand">CleanKitchen NYC</div>', unsafe_allow_html=True)
    if st.button("", key="nav_brand"):
        navigate_to('home')  # Navigates to home page

with col2:
    # Using columns for better alignment and styling control with st.button
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 1])

    # The actual links are now buttons styled to look like text
    with nav_col1:
        if st.button("About", key="nav_about"):
            navigate_to('about') # Navigates to a dummy 'about' page
    with nav_col2:
        if st.button("Predictor", key="nav_services"):
            # SERVICES links to the PREDICTION PAGE
            navigate_to('prediction')
    with nav_col3:
        if st.button("Filter", key="nav_filter"):
            navigate_to('filter') # Navigates to a dummy 'portfolio' page
    with nav_col4:
        if st.button("Blog", key="nav_blog"):
            navigate_to('blog') # Navigates to a dummy 'blog' page

with col3:
    st.button("Contact us", key="nav_contact") # Contact us stays as a normal button

# --- The Horizontal Divider (Always rendered) ---
st.markdown('<div class="header-separator"></div>', unsafe_allow_html=True)

# --- Content Rendering Logic ---

"""
    Renders the custom input form and uses the model to predict the grade 
    for a hypothetical, user-defined restaurant instance.
    """
def prediction_page():

    st.title(" Predict Next Inspection Grade")
    st.subheader("Input Restaurant Data")
    
    # -------------------------------------------------
    # USER INPUT FORM (Centered layout)
    # -------------------------------------------------
    
    # Use columns to center the input fields
    _, input_col, _ = st.columns([1, 6, 1])

    with input_col:
        st.markdown("---")
        
        # We need options for Borough and Cuisine from the existing data
        boroughs = sorted(df["borough"].dropna().unique().tolist())
        cuisine_list = sorted(df["cuisine_description"].dropna().unique().tolist())
        
        # Row 1: Location and Name
        col1, col2 = st.columns(2)
        with col1:
            # 1. Name/DBA (for display only)
            dba_input = st.text_input("Restaurant Name", "Hypothetical Eatery")
            # 2. Borough (Categorical input)
            borough_choice = st.selectbox("Borough", boroughs, index=boroughs.index('Manhattan') if 'Manhattan' in boroughs else 0)
        with col2:
            # 3. ZIP Code (Numerical/Categorical input)
            zipcode_input = st.text_input("ZIP Code", "10001")
            # 4. Cuisine (Categorical input)
            cuisine_choice = st.selectbox("Cuisine Type", cuisine_list, index=cuisine_list.index('American') if 'American' in cuisine_list else 0)

        # Row 2: Inspection Status (Actionable features used in the ML model)
        st.markdown("### Latest Inspection Status *(If Applicable)*")
        col3, col4 = st.columns(2)
        with col3:
            # 5. Last Inspection Score (Numerical input)
            # This is a critical feature often used for prediction
            score_input = st.number_input(
                "Last Inspection Score", 
                min_value=0, 
                max_value=100, 
                value=10, 
                help="A lower score is better (A: 0-13, B: 14-27, C: 28+)"
            )
        with col4:
            # 6. Days Since Last Inspection (Numerical/Temporal feature)
            # This is essential for predicting the likelihood of the *next* inspection.
            days_since_input = st.number_input(
                "Days Since Last Inspection", 
                min_value=0, 
                max_value=730, # Max two years
                value=100, 
                help="Enter the number of days since the restaurant was last inspected."
            )
            
        st.markdown("---")

        if st.button("Predict Next Inspection Grade", width='stretch'):
            with st.spinner(f"Analyzing {dba_input} data..."):
                
                # -----------------------------------------------------------------
                # STEP 1: CONSTRUCT THE MOCK INPUT ROW
                # The model requires a row (pd.Series) with specific column names.
                # We need to map user inputs to the feature names the model expects.
                # NOTE: This only works if your 'row_to_model_input' function can 
                # handle a minimally populated Series/DataFrame row.
                # -----------------------------------------------------------------
                
                mock_input_data = {
                    'DBA': dba_input, # Display only
                    'dba': dba_input, # Display only
                    'borough': borough_choice,
                    'zipcode': zipcode_input,
                    'cuisine_description': cuisine_choice,
                    'score': score_input,
                    # We assume the model expects this explicit feature:
                    'days_since_last_inspection': days_since_input, 
                    
                    # Add placeholders for other fields needed by the feature pipeline 
                    # (e.g., date, grade, camis are often required by 'row_to_model_input')
                    'inspection_date': datetime.now() - pd.Timedelta(days=days_since_input),
                    'grade': 'Z', # Placeholder for prediction target
                    'camis': '00000000',
                    'latitude': 40.7, # Dummy location
                    'longitude': -74.0, # Dummy location
                    'street': 'User Input'
                }
                
                # Create a single-row DataFrame (or Series)
                selected_row = pd.Series(mock_input_data)


                try:
                    # -----------------------------------------------------------------
                    # STEP 2: PREDICT GRADE (Algorithm Pipelining Kept Intact)
                    # -----------------------------------------------------------------
                    
                    # Build model input (this is where feature engineering happens implicitly)
                    model_input = row_to_model_input(selected_row)
                    result = predict_restaurant_grade(model_input)

                    predicted_grade = result["grade"]
                    probabilities = result["probabilities"]
                    formatted_probs = format_probabilities(probabilities)

                    # Get current inspection info (from user input for status checks)
                    current_score = score_input
                    
                    # ... [Derived Grade, Time Estimate, and Display Logic remains the same] ...
                    
                    # --- DERIVED STATUS LOGIC (Copied from original) ---
                    # Derive grade from score for status info
                    derived_grade = None
                    if current_score <= 13:
                        derived_grade = 'A'
                    elif current_score <= 27:
                        derived_grade = 'B'
                    else:
                        derived_grade = 'C'
                    grade_status = f"{derived_grade} (Current Status)"
                    
                    # Calculate expected time to next inspection
                    days_since = days_since_input
                    median_interval = 124 
                    
                    if days_since > 365:
                        years_ago = round(days_since / 365, 1)
                        time_estimate = f"Last inspected {years_ago:.0f}+ years ago"
                    elif days_since > median_interval:
                        time_estimate = "Overdue for inspection"
                    elif days_since > median_interval - 30:
                        time_estimate = "Due within ~1 month"
                    elif days_since > median_interval - 60:
                        time_estimate = "Expected in ~2 months"
                    elif days_since > median_interval - 90:
                        time_estimate = "Expected in ~3 months"
                    else:
                        months = round((median_interval - days_since) / 30)
                        time_estimate = f"Expected in ~{months} months"

                    # --- DISPLAY RESULTS ---
                    pred_color = get_grade_color(predicted_grade)
                    
                    st.markdown("### Prediction Results")

                    # Current Status Card (based on user score)
                    st.markdown(f"""
                    <div class="info-card" style="text-align: center;">
                        <p style="font-size: 0.85rem; margin-bottom: 0.5rem; color: #6C757D;">
                            CURRENT DERIVED GRADE STATUS (Score: {current_score})
                        </p>
                        <div class="grade-badge grade-{derived_grade}" style="margin: 0 auto; background: {get_grade_color(derived_grade)};">
                            {derived_grade}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Next inspection prediction card
                    st.markdown(f"""
                    <div class="info-card" style="text-align: center; border: 2px solid {pred_color}; margin-top: 1rem;">
                        <p style="font-size: 0.75rem; margin-bottom: 0.5rem; color: #6C757D; text-transform: uppercase;">
                            Predicted Grade on Next Inspection
                        </p>
                        <p style="font-size: 0.7rem; color: #888; margin-bottom: 8px;">
                            {time_estimate}
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

                    # Historical context
                    if derived_grade == 'C' or current_score >= 28:
                        st.markdown("""
                        <div style="background: #F8F9FA; padding: 12px; border-radius: 8px; margin-top: 12px; font-size: 0.8rem; color: #6C757D;">
                            <strong>Historical Pattern:</strong> 64% of restaurants that fail an inspection
                            pass their re-inspection within ~4 months after addressing violations.
                        </div>
                        """, unsafe_allow_html=True)

                except ModelNeedsRetrainingError:
                    st.error("Model needs retraining. Please check the **Data Management** page.")
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

    
    st.markdown("---")
    st.markdown("#### Management Actions (Kept for functionality)")
    col_data, col_model = st.columns(2)
    with col_data:
        if st.button("🔄 Fetch Fresh Data"):
             # Original data fetch logic would go here
             pass
    with col_model:
        if st.button("🧠 Retrain Model"):
             # Original retraining logic would go here
             pass

    # Note: The original logic for model training and data fetching is complex. 
    # For a final app, these blocks should be placed on a separate "Data Management" page 
    # as they cause the app to rerun and clutter the main prediction interface.

def home_page():
    """Renders the content for the Home Page (your original design)."""
    # --- Hero Section ---
    st.markdown('<div class="hero-sub">Health Predicting/Filtering • NYC</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Healthier Food<br>for NYC</div>', unsafe_allow_html=True)

    # "Get a free quote" button with pill border - NOW NAVIGATES
    b_col, _ = st.columns([2, 8])
    with b_col:
        # GET A FREE QUOTE button links to the PREDICTION PAGE
        if st.button("Get Started →", key="hero_quote"):
            navigate_to('prediction')

    st.markdown("<br><br>", unsafe_allow_html=True)
    image_links = [
        "https://picsum.photos/id/63/900/600",
        "https://picsum.photos/id/292/900/600",
        "https://picsum.photos/id/425/900/600",
        "https://picsum.photos/id/429/900/600",
        "https://picsum.photos/id/488/900/600",
        "https://picsum.photos/id/493/900/600"
    ]

    # Create the HTML structure for one set of items
    content_items_html = ""
    for url in image_links:
        # Use the <img> tag directly within the "marquee-item" div
        content_items_html += f"""
        <div class="marquee-item">
            <img src="{url}" alt="Marquee Image">
        </div>
        """

    # To make the loop seamless, the content is duplicated
    looped_content = content_items_html * 2


    # --- 3. Render the Marquee Component ---
    st.markdown(
        f"""
        <div class="marquee-container">
            <div class="marquee-content">
                {looped_content}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
def filter_page():
    
    st.initial_sidebar_state="expanded"

    if df.empty:
        st.error("No data loaded. Please check your CSV files in the data/ folder.")
        st.stop()


    # -------------------------------------------------
    # SIDEBAR FILTERS
    # -------------------------------------------------
    st.sidebar.header(" Filter Restaurants")

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
                radius_min_pixels=4,
                radius_max_pixels=8,
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
                width='stretch',
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
                width='stretch'
            )
        else:
            st.dataframe(df_filtered.head(300), width='stretch')


    with right_col:
        st.subheader("Restaurant Details")

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

            # Build labels with street name and ZIP for easy identification
            def make_label(i):
                name = df_filtered.loc[i, name_col]
                street = df_filtered.loc[i, 'street'] if 'street' in df_filtered.columns else None
                zipcode = df_filtered.loc[i, 'zipcode']

                if pd.notna(street) and street not in ('nan', ''):
                    return f"{name} ({street}, {zipcode})"
                else:
                    # Fall back to borough if no street
                    borough = df_filtered.loc[i, 'borough']
                    return f"{name} ({borough}, {zipcode})"

            labels = [make_label(i) for i in options]

            selected_idx = st.selectbox(
                "Choose a restaurant :",
                options=options,
                format_func=lambda i: labels[i]
            )

            selected_row = df_filtered.loc[selected_idx]

            # Get values
            restaurant_name = selected_row.get(name_col, 'N/A')
            borough = selected_row.get('borough', 'N/A')
            zipcode = selected_row.get('zipcode', 'N/A')
            cuisine = selected_row.get('cuisine_description', 'N/A')
            score = display_value(selected_row.get('score'), 'N/A')
            grade = display_value(selected_row.get('grade'), 'N/A')
            grade_color = get_grade_color(grade if grade != 'N/A' else 'Z')

            # Format inspection date nicely
            inspection_date = selected_row.get('inspection_date')
            if pd.notna(inspection_date):
                try:
                    inspection_date_str = pd.to_datetime(inspection_date).strftime('%b %d, %Y')
                except:
                    inspection_date_str = str(inspection_date)
            else:
                inspection_date_str = 'N/A'

            # Restaurant info card
            st.markdown(f"""
            <div class="info-card">
                <h3 style="margin: 0 0 12px 0; font-size: 1.1rem; color: #2C3E50;">{restaurant_name}</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.9rem;">
                    <div>
                        <span style="color: #6C757D;">Borough</span><br/>
                        <span style="font-weight: 500;">{borough}</span>
                    </div>
                    <div>
                        <span style="color: #6C757D;">ZIP</span><br/>
                        <span style="font-weight: 500;">{zipcode}</span>
                    </div>
                    <div style="grid-column: span 2;">
                        <span style="color: #6C757D;">Cuisine</span><br/>
                        <span style="font-weight: 500;">{cuisine}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Latest inspection card
            st.markdown(f"""
            <div class="info-card">
                <h4 style="margin: 0 0 12px 0; font-size: 0.95rem; color: #6C757D;">Latest Inspection</h4>
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div style="text-align: center;">
                        <div class="grade-badge" style="background: {grade_color}; width: 48px; height: 48px; font-size: 1.3rem;">
                            {grade}
                        </div>
                        <div style="font-size: 0.75rem; color: #6C757D; margin-top: 4px;">Grade</div>
                    </div>
                    <div style="flex: 1; font-size: 0.9rem;">
                        <div style="margin-bottom: 6px;">
                            <span style="color: #6C757D;">Score:</span>
                            <span style="font-weight: 600; margin-left: 4px;">{score}</span>
                        </div>
                        <div>
                            <span style="color: #6C757D;">Date:</span>
                            <span style="font-weight: 500; margin-left: 4px;">{inspection_date_str}</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Inspection History (last 5 inspections)
            camis = selected_row.get('camis')
            if camis:
                raw_df = get_raw_data()
                history = raw_df[raw_df['camis'] == camis].copy()
                history = history.sort_values('inspection_date', ascending=False)
                # Get unique inspections by date
                history = history.drop_duplicates(subset=['inspection_date']).head(5)

                if len(history) > 0:
                    history_items = []
                    for _, insp in history.iterrows():
                        insp_date = insp.get('inspection_date')
                        if pd.notna(insp_date):
                            try:
                                date_str = pd.to_datetime(insp_date).strftime('%b %d, %Y')
                            except:
                                date_str = str(insp_date)
                        else:
                            date_str = 'N/A'

                        insp_grade = display_value(insp.get('grade'), '-')
                        insp_score = display_value(insp.get('score'), '-')
                        insp_color = get_grade_color(insp_grade if insp_grade != '-' else 'Z')

                        history_items.append(
                            f'<div style="display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid #eee;">'
                            f'<div class="grade-badge" style="background: {insp_color}; width: 32px; height: 32px; font-size: 0.9rem;">{insp_grade}</div>'
                            f'<div style="flex: 1; font-size: 0.85rem;">'
                            f'<span style="font-weight: 500;">{date_str}</span>'
                            f'<span style="color: #6C757D; margin-left: 8px;">Score: {insp_score}</span>'
                            f'</div></div>'
                        )

                    history_html = (
                        '<div class="info-card">'
                        '<h4 style="margin: 0 0 12px 0; font-size: 0.95rem; color: #6C757D;">Inspection History</h4>'
                        + ''.join(history_items) +
                        '</div>'
                    )
                    st.markdown(history_html, unsafe_allow_html=True)


def blog_page():
    """Renders the content for the Blog Page."""
    st.title("💡 Blog Page")
    st.write("Welcome to the blog page! Here you can read the latest blog posts.")
    st.markdown("---")
    st.markdown("### Placeholder for your Blog Page")
    st.info("You would add your blog posts here.")
    if st.button("← Back to Home"):
        navigate_to('home')


# --- Page Router ---
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'prediction':
    prediction_page()
elif st.session_state.page == 'filter':
    filter_page()
elif st.session_state.page == 'blog':
    blog_page()
else:
    # Handles clicks on About, Portfolio, Blog with a simple placeholder
    if st.button("← Go Home"):
        navigate_to('home')
        