import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import datetime
import matplotlib.pyplot as plt


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
        padding-top: 10
        px;
    }

    /* Change nav-link to be a clickable button-like element */
    .nav-link {
        font-family: sans-serif;
        font-size: 20px;
        font-weight: 600;
        text-transform: uppercase;
        color: #333;
        text-decoration: none;
        letter-spacing: 0.5px;
        cursor: pointer;
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
/*--------------- Team Member Card Styles ---------------*/
.team-member-container {
        display: flex;
        align-items: center; /* Vertically center image and text block */
        gap: 30px;
        max-width: 750px;
        margin: 30px auto; /* Center the member block */
    }
    .profile-img {
        width: 200px; 
        height: 200px;
        border-radius: 50%; /* Makes the image circular */
        object-fit: cover;
        flex-shrink: 0; /* Prevents the image from shrinking */
        border: 5px solid #E0E0E0; /* Light grey border */
    }
    .member-details {
        flex-grow: 1;
        text-align: left;
    }
    .member-details strong {
        font-size: 2.5rem;
        font-weight: 800;
        display: block;
        margin-bottom: 5px;
    }
    .member-details ul {
        list-style: none; /* Remove default bullet points */
        padding: 0;
        margin: 0;
        font-size: 1rem;
        line-height: 1.6;
    }
    .linkedin-icon {
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 5px;
        vertical-align: middle;
    }
    .github-icon {
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 5px;
        vertical-align: middle;
    .divider {
        border-bottom: 1px solid #DDDDDD; /* Light grey line */
        margin: 40px auto;
        width: 80%;
    }
    /*--------------- About Body ---------------*/
    /* Newspaper Body: Two-column layout effect */
    .newspaper-body {
        column-count: 2; /* Main feature of the newspaper look */
        column-gap: 30px; /* Space between columns */
        text-align: justify; /* Justified text */
        line-height: 1.5;
        font-family: 'Times New Roman', Times, serif;
        font-size: 1.05rem;
    }

    /* Ensure list items span columns correctly */
    .newspaper-body ul, .newspaper-body p {
        /* Prevents bullet points from breaking across columns too easily */
        -webkit-column-break-inside: avoid;
        break-inside: avoid;
    }

    /* Clear the column layout for titles/headers */
    .column-clear {
        column-span: all;
        margin-bottom: 15px;
        margin-top: 10px;
        border-bottom: 1px solid #CCCCCC;
    }
    
    /* Ensure the image doesn't break the columns */
    .stImage, .stVideo, .stPlot {
        column-span: all;
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
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])

    # The actual links are now buttons styled to look like text
    with nav_col1:
        if st.button("About", key="nav_about"):
            navigate_to('about') # Navigates to a dummy 'about' page
    with nav_col2:
        if st.button("Filter/Predictor", key="nav_filter"):
            navigate_to('filter') # Navigates to a dummy 'portfolio' page
    with nav_col3:
        if st.button("Blog", key="nav_blog"):
            navigate_to('blog') # Navigates to a dummy 'blog' page

with col3:
    st.button("Contact us", key="nav_contact") # Contact us stays as a normal button

# --- The Horizontal Divider (Always rendered) ---
st.markdown('<div class="header-separator"></div>', unsafe_allow_html=True)


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
    
    st.sidebar.header("Filter Restaurants")

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
    # Model Management (sidebar)
    # -------------------------------------------------
    st.sidebar.divider()
    st.sidebar.caption("Model Management")

    # Show current model info
    try:
        metadata = get_model_metadata()
        training_date = metadata.get("training_date", "Unknown")
        if training_date != "Unknown":
            # Parse ISO format date
            try:
                dt = datetime.fromisoformat(training_date)
                training_date = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
        metrics = metadata.get("training_metrics", {})
        accuracy = metrics.get("accuracy")
        if accuracy:
            st.sidebar.markdown(f"**Last trained:** {training_date}")
            st.sidebar.markdown(f"**Accuracy:** {accuracy:.1%}")
        else:
            st.sidebar.markdown("**Model:** Not yet trained with new features")
    except Exception:
        st.sidebar.markdown("**Model:** Ready for training")

    # Training button
    if st.sidebar.button("🧠 Retrain Model"):
        with st.sidebar:
            progress_bar = st.progress(0, text="Loading training data...")

            try:
                # Step 1: Load raw data
                progress_bar.progress(10, text="Loading raw inspection data...")
                raw_df = load_training_data()

                # Step 2: Compute features (using training-specific function for correct temporal alignment)
                progress_bar.progress(30, text="Computing training features...")
                feature_df = compute_all_features(raw_df)

                # Step 3: Train model
                progress_bar.progress(50, text="Training model...")
                model, metrics, feature_importances, encoders = train_model(feature_df)

                # Step 4: Save model
                progress_bar.progress(80, text="Saving model...")
                save_model(model, encoders, metrics, feature_importances)

                # Step 5: Clear caches
                progress_bar.progress(90, text="Clearing caches...")
                clear_model_cache()
                clear_data_cache()

                progress_bar.progress(100, text="Complete!")

                # Show results
                st.success("Model trained successfully!")
                st.markdown(f"""
                **Training Results:**
                - Accuracy: {metrics['accuracy']:.1%}
                - Precision: {metrics['precision']:.1%}
                - Recall: {metrics['recall']:.1%}
                - F1 Score: {metrics['f1']:.1%}
                - Training samples: {metrics['train_samples']:,}
                """)

                # Show top feature importances
                st.markdown("**Top Features:**")
                top_features = get_feature_importance_ranking(feature_importances)[:5]
                for name, importance in top_features:
                    st.markdown(f"- {name}: {importance:.3f}")

            except Exception as e:
                st.error(f"Training failed: {e}")

    st.sidebar.caption("Training takes ~30 seconds")


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
                "Choose a restaurant to analyze:",
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

            # Check if model needs retraining
            if model_needs_retraining():
                st.warning(
                    "**Model needs to be trained.** The prediction model has been updated with new features. "
                    "Please click **'Retrain Model'** in the sidebar to train the model before making predictions."
                )

            if st.button("Predict Next Inspection", width='stretch'):
                with st.spinner("Analyzing restaurant data..."):
                    try:
                        # Build model input
                        model_input = row_to_model_input(selected_row)
                        result = predict_restaurant_grade(model_input)

                        predicted_grade = result["grade"]
                        probabilities = result["probabilities"]
                        formatted_probs = format_probabilities(probabilities)

                        # Get current inspection info
                        current_score = selected_row.get('score')
                        current_grade = selected_row.get('grade')

                        # Derive grade from score if official grade not available
                        if pd.isna(current_grade) or current_grade not in ['A', 'B', 'C']:
                            if pd.notna(current_score):
                                if current_score <= 13:
                                    derived_grade = 'A'
                                elif current_score <= 27:
                                    derived_grade = 'B'
                                else:
                                    derived_grade = 'C'
                                grade_status = f"{derived_grade} (pending)"
                            else:
                                derived_grade = None
                                grade_status = "Pending"
                        else:
                            derived_grade = current_grade
                            grade_status = current_grade

                        # Calculate expected time to next inspection
                        days_since = selected_row.get('days_since_last_inspection', 0)
                        if pd.isna(days_since):
                            days_since = 0

                        median_interval = 124  # median days between inspections

                        # Handle stale data (no inspection in over a year)
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

                        # Next inspection prediction card
                        pred_color = get_grade_color(predicted_grade)
                        st.markdown(f"""
                        <div class="info-card" style="text-align: center; border: 2px solid {pred_color};">
                            <p style="font-size: 0.75rem; margin-bottom: 0.5rem; color: #6C757D; text-transform: uppercase;">
                                Predicted Next Inspection
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
                        if derived_grade == 'C' or (pd.notna(current_score) and current_score >= 28):
                            st.markdown("""
                            <div style="background: #F8F9FA; padding: 12px; border-radius: 8px; margin-top: 12px; font-size: 0.8rem; color: #6C757D;">
                                <strong>Historical Pattern:</strong> 64% of restaurants that fail an inspection
                                pass their re-inspection within ~4 months after addressing violations.
                            </div>
                            """, unsafe_allow_html=True)

                    except ModelNeedsRetrainingError:
                        st.error(
                            "**Model needs retraining.** Please click 'Retrain Model' in the sidebar first."
                        )
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
                                      
def render_member_card(name, college, major, linkedin_url, image_url, github_url):
    """Generates the HTML/Markdown for a single team member card."""
    
    # Placeholder image for the LinkedIn icon
    linkedin_icon_url = "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"
    linkedin_icon_html = f'<img class="linkedin-icon" src="{linkedin_icon_url}" alt="LinkedIn Icon">'
    github_icon_url = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"   
    github_icon_html = f'<img class="github-icon" src="{github_icon_url}" alt="GitHub Icon">'

    html = f"""
    <div class="team-member-container">
        <div>
            <img class="profile-img" src="{image_url}" alt="{name}'s Photo">
        </div>
        <div class="member-details">
            <strong>{name}</strong>
            <ul>
                <li>College: {college}</li>
                <li>Major: {major}</li>
                <li>LinkedIn: {linkedin_icon_html} <a href="{linkedin_url}" target="_blank">{name}</a></li>
                <li>GitHub: {github_icon_html} <a href="{github_url}" target="_blank">{name}</a></li>
            </ul>
        </div>
    </div>
    <div class="divider"></div>
    """
    st.markdown(html, unsafe_allow_html=True)

def blog_page():
    st.markdown("<h1 style='text-align: center; font-size: 75px;'>Creators</h1>", unsafe_allow_html=True)
    st.markdown("---")
    # --- MEMBER 1: MANUALLY EDIT THIS BLOCK ---
    render_member_card(
        name="Jack Kaplan",
        college="CUNY College",
        major="Computer Science",
        linkedin_url="https://www.linkedin.com/in/jackkaplan1",
        github_url="https://github.com/Jack-Kaplan",
        image_url="https://ca.slack-edge.com/T094PKG3ASD-U095B5FJP1P-d40df1de134c-512" 
        
    )
    st.markdown("---")

    # --- MEMBER 2: MANUALLY EDIT THIS BLOCK ---
    render_member_card(
        name="Dominik Kasza",
        college="Queens College",
        major="[TEAMMATE MAJOR/FIELD OF STUDY]",
        linkedin_url="https://www.linkedin.com/in/dominik-kasza-",
        github_url="https://github.com/Guciosk",
        image_url="https://ca.slack-edge.com/T094PKG3ASD-U09628KG754-788280e58034-512" 
    )
    st.markdown("---")

    # --- MEMBER 3: MANUALLY EDIT THIS BLOCK ---
    render_member_card(
        name="Mauricio",
        college="CUNY College",
        major="Computer Science",
        linkedin_url="",
        github_url="https://github.com/M4URIC18",
        image_url="https://ca.slack-edge.com/T094PKG3ASD-U096M8URFAR-dfe4d20b491f-512"
    )
    
    st.markdown("---")

    if st.button("End of Team Page"):
        st.info("You've reached the end of the manual input section.")

def about_page():
    
    st.title("CleanKitchen NYC Project Analysis")
    
    # Wrap the entire content section in the newspaper body container
    st.markdown('<div class="newspaper-body">', unsafe_allow_html=True)
    
    # -------------------------------------------------
    #  Project Goal
    # -------------------------------------------------
    # Use h2 style for major sections, setting column-span=all for the header
    st.markdown('<div class="column-clear"></div>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-size: 1.75rem;"> Project Goal</h2>', unsafe_allow_html=True)
    
    st.markdown(
        """
        Each year, the **New York City Health Department** inspects roughly 24,000 restaurants and evaluates them on food handling, temperature control, hygiene, and vermin management. To help the public understand inspection results, the city introduced a letter-grade system in 2010, assigning each restaurant an **A, B, or C**, with **A** being the highest score. Restaurants must display this grade at their entrance so customers can easily gauge their health standards. 
        """
    )

    # Image placement—this spans both columns
    st.markdown(
        f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="https://picsum.photos/id/1/600/300" alt="NYC Health Grades" style="max-width: 100%; height: auto;">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Because restaurants in NYC often change ownership or reopen, many display **“Grade Pending”** or **“Not Yet Graded.”** In these cases, customers can only rely on basic information—such as the restaurant’s name, cuisine, address, borough, and ZIP code—to guess what grade it might eventually receive.
        
        The goal of this project is to use **only these publicly visible attributes** to predict whether a restaurant will earn an A or a lower grade.
        """
    )

    # -------------------------------------------------
    #  Data
    # -------------------------------------------------
    st.markdown('<h2 style="font-size: 1.75rem;"> Data</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        NYC publishes all restaurant inspection results through its public database on **NYC Open Data**. These downloadable text files include:
        
        * Restaurant name, address, borough, ZIP code
        * Cuisine type
        * Inspection dates
        * Violations observed
        * Inspection score and assigned grade
        * Any enforcement actions
        
        Because this project aims to mimic what a typical customer would know, **only public-facing attributes are used as model inputs.**
        """
    )
    
    st.markdown('<div class="column-clear"></div>', unsafe_allow_html=True)
    st.subheader("Data Preparation Steps")
    st.markdown(
        """
        To prepare the dataset:
        
        * The text files were converted into a spreadsheet format for analysis in MATLAB.
        * Inspections without a final grade were removed.
        * Restaurants labeled “Grade Pending” or “Not Yet Graded” were excluded.
        * Because many restaurants undergo multiple inspections, grades were averaged to create a single "typical" score per restaurant.
        * The dataset was split: **75% training data, 25% test data.**
        
        After cleaning, **2,768 unique graded restaurants remained**. About 52.6% received an A, while the rest earned lower grades.
        """
    )
    
    # -------------------------------------------------
    #  Approach
    # -------------------------------------------------
    st.markdown('<h2 style="font-size: 1.75rem;"> Approach</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        The approach is to evaluate each feature individually—based solely on information available to a customer—and estimate the **probability** that a restaurant receives an A based on that single feature. These probabilities are then combined using **logistic regression** to produce a final prediction.
        
        To support effective feature selection:
        
        * Each feature is tested independently with its own held-out test set.
        * The model output for each feature is simply the **predicted probability of an A.**
        
        Two important considerations arise:
        
        1.  Features fall into different categories: **Textual** (name, street address), **Independent categorical** (borough, cuisine), and **Ordered/related numeric** (ZIP code).
        2.  Some features are strongly correlated—e.g., ZIP code ↔ borough, or name ↔ cuisine—which limits which features can be combined without harming performance. **Models avoid using pairs of features that are clearly correlated.**
        """
    )
    
    # -------------------------------------------------
    #  Feature Analyses
    # -------------------------------------------------
    st.markdown('<h2 style="font-size: 1.75rem;"> Feature Analyses</h2>', unsafe_allow_html=True)

    st.subheader("Borough")
    st.markdown(
        """
        Using **Bayes’ Rule** with indicator functions and evaluating with a Naïve Bayes classifier:
        
        * Staten Island alone achieves **74.2% accuracy**.
        * Citywide accuracy is **51.3%**, effectively random.
        
        This suggests borough is too coarse—boroughs are large and diverse. Staten Island performs better mainly because it has a more homogeneous population.
        """
    )
    
    st.subheader("Food Type")
    st.markdown(
        """
        Using Naïve Bayes on cuisine type yields **61.4% accuracy**, significantly better than borough.
        
        * **Strong predictors** (generalization error < 30%): Asian (Chinese/Japanese), Donuts, Greek, Ice Cream, Indian, Indonesian, Juice/Smoothie, Russian, Sandwiches, Steak, Turkish.
        * **Weak predictors** (error > 45%): African, Bagels, Bakery, Italian, Korean, Kosher, Middle Eastern, Pizza, Spanish, Vegetarian.
        
        Food type is therefore a valuable feature.
        """
    )

    st.subheader("Name")
    st.markdown(
        """
        Restaurant names were evaluated using a **Naïve Bayes text classifier** similar to standard spam filters.
        
        * Accuracy: **58.3%**, statistically significant.
        * Training set produced 2,948 unique words.
        * Words strongly associated with grade predictions include: **Bombay, Chen, Dunkin, Donuts, Fusion, Ice, Pain, Quotidien, Shoppe, Town.**
        
        Many of these correlate with cuisine type (e.g., “Donuts,” “Bombay,” “Ice”). Because of this overlap, **name and cuisine should not be combined** in the final model.
        """
    )

    st.subheader("Street Address")
    st.markdown(
        """
        Using text classification on street names:
        
        * 656 unique street terms.
        * Accuracy: **52.5%**.
        
        Although certain streets (e.g., Richmond, Flatlands, Pearl, Knickerbocker) correlate with specific grades, overall performance is weak. **Street name is not a reliable feature.**
        """
    )
    
    st.subheader("ZIP Code")
    st.markdown(
        """
        ZIP code is more structured because nearby ZIP codes share geographic boundaries. Three approaches were tested:
        
        1.  **Naïve Bayes:** Accuracy: 55.7% across 188 ZIP codes. Statistically meaningful but weaker than name or food type.
        2.  **k-means Clustering:** ZIP codes grouped into 42 clusters (matching NYC’s 42 recognized neighborhoods). Accuracy: 53%, slightly worse than Naïve Bayes.
        3.  **Polynomial Fitting:**
            * Group ZIP codes into seven natural numeric clusters (e.g., 103xx → Staten Island).
            * Encode A = 1, else = 0.
            * Use cross-validation to find optimal polynomial degree.
            * Fit a polynomial to each group to estimate probabilities.
            * Accuracy: 53.1%, slightly above random.
            
        While polynomial fitting captures adjacency patterns, it performs worse than Naïve Bayes when used alone.
        """
    )
    
    # -------------------------------------------------
    #  Combining Features
    # -------------------------------------------------
    st.markdown('<h2 style="font-size: 1.75rem;"> Combining Features</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        To improve prediction, the strongest independent features are combined using **logistic regression**:
        
        * **Inputs:** Probability based on **food type** and Naïve Bayes probability from **ZIP code**.
        * Training uses gradient descent with a decreasing learning rate until convergence.
        """
    )

    # Use a standard Streamlit column layout for the final results, 
    # breaking the CSS column flow for the final presentation.
    st.markdown('</div>', unsafe_allow_html=True) # CLOSE newspaper-body DIV
    st.markdown("---") # Visual separator

    col_res, col_why = st.columns(2)
    
    with col_res:
        st.subheader("Final Results")
        st.markdown(
            """
            * Logistic regression with food type + Naïve Bayes ZIP $\approx$ similar accuracy to food type alone.
            * Logistic regression with food type + **polynomial ZIP code** improves accuracy to **62.4%**.
            """
        )

    with col_why:
        st.subheader("Key Insight")
        st.markdown(
            """
            **Why polynomial ZIP works better in combination:**
            
            The polynomial estimate may be **biased**, but logistic regression corrects this during training through its learned coefficients. This improvement is consistent across multiple train/test splits, showing that the combination of Food Type's strong inherent prediction with the structurally corrected prediction from Polynomial ZIP yields the best final result.
            """
        )
    
    st.markdown("---")
    
    # Placeholder for navigation or next step
    if st.button("Back to Home"):
        # Assuming you have a navigation function like navigate_to('home')
        pass
# --- Page Router ---
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'filter':
    filter_page()
elif st.session_state.page == 'blog':
    blog_page()
elif st.session_state.page == 'about':
    about_page()    
else:
    # Handles clicks on About, Portfolio, Blog with a simple placeholder
    if st.button("← Go Home"):
        navigate_to('home')
        