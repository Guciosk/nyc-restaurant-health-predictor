import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import datetime
import pages.filter as filter
import pages.prediction as prediction
import pages.blog as blog   
import pages.about as about

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
    .brand  {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 50px;
        font-weight: 800;
        color: #000;
        text-decoration: none;
        border: none;
        text-border: none !important;
        oncursor: pointer;
    }
    
    /* 💥 BRAND BUTTON FIX: Container for the brand text and button */
    #brand_link_container {
        position: relative;
        cursor: pointer; 
        padding-bottom: 5px;
        display: inline; /* Crucial to wrap the text closely */
        
    }

    
    #brand_link_container button {
        background-color: transparent !important;
        color: transparent !important; /* Hide button text */
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        /* Make the hidden button cover the large brand text area */
        width: 50%; 
        height: 100%;
        position: absolute; 
        top: 0;
        left: 0;
        z-index: 10;
        font-size: 0 !important; /* Fully hide any text */
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
    
    .title-button > brand {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 50px;
        font-weight: 800;
        color: #000;
        text-decoration: none;
        border: none;
        text-border: none !important;
        oncursor: pointer;
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
    # 1. Create a container for the custom brand link
    st.markdown('<div id="brand_link_container">', unsafe_allow_html=True)
    
    # 2. Render the actual large brand text
    st.markdown('<div class="brand">CleanKitchen NYC</div>', unsafe_allow_html=True)
    
    # 3. Overlay the hidden Streamlit button to capture the click action
    if st.button(" ", key="brand_home_button"): # Use a space here to help the button render
        navigate_to('home')
        
    st.markdown('</div>', unsafe_allow_html=True)

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

    # --- Image Grid ---
    c1, c2, c3, c4 = st.columns(4)

    def render_image(col, label):
        with col:
            # REPLACE THIS BLOCK WITH st.image("path.jpg") WHEN READY
            st.markdown(f'<div class="img-placeholder">{label}</div>', unsafe_allow_html=True)

    render_image(c1, "IMG1<br>Placeholder")
    render_image(c2, "IMG2<br>Placeholder")
    render_image(c3, "IMG3<br>Placeholder")
    render_image(c4, "IMG4<br>Placeholder")



# --- Page Router ---
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'prediction':
    prediction.prediction_page()
elif st.session_state.page == 'filter':
    filter.filter_page()
elif st.session_state.page == 'blog':
    blog.blog_page()
elif st.session_state.page == 'about':
    about.about_page()  
else:
    # Handles clicks on About, Portfolio, Blog with a simple placeholder
    if st.button("← Go Home"):
        navigate_to('home')
        
