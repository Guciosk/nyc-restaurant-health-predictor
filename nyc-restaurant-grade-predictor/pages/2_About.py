"""
About Page - Project documentation and methodology for CleanKitchen NYC.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.components import load_css, render_header_divider

# --- Page Configuration ---
st.set_page_config(
    page_title="About - CleanKitchen NYC",
    layout="wide",
    initial_sidebar_state="collapsed"
)
load_css()

# --- Page Content ---
st.title("CleanKitchen NYC Project Analysis")
render_header_divider()

# Wrap the content section in the newspaper body container
st.markdown('<div class="newspaper-body">', unsafe_allow_html=True)

# -------------------------------------------------
#  Project Goal
# -------------------------------------------------
st.markdown('<div class="column-clear"></div>', unsafe_allow_html=True)
st.markdown('<h2 style="font-size: 1.75rem;">Project Goal</h2>', unsafe_allow_html=True)

st.markdown(
    """
    Each year, the **New York City Health Department** inspects roughly 24,000 restaurants and evaluates them on food handling, temperature control, hygiene, and vermin management. To help the public understand inspection results, the city introduced a letter-grade system in 2010, assigning each restaurant an **A, B, or C**, with **A** being the highest score. Restaurants must display this grade at their entrance so customers can easily gauge their health standards.
    """
)

st.markdown(
    """
    Because restaurants in NYC often change ownership or reopen, many display **"Grade Pending"** or **"Not Yet Graded."** In these cases, customers can only rely on basic information--such as the restaurant's name, cuisine, address, borough, and ZIP code--to guess what grade it might eventually receive.

    The goal of this project is to use **only these publicly visible attributes** to predict whether a restaurant will earn an A or a lower grade.
    """
)

# -------------------------------------------------
#  Data
# -------------------------------------------------
st.markdown('<h2 style="font-size: 1.75rem;">Data</h2>', unsafe_allow_html=True)
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

    * The text files were converted into a spreadsheet format for analysis.
    * Inspections without a final grade were removed.
    * Restaurants labeled "Grade Pending" or "Not Yet Graded" were excluded.
    * Because many restaurants undergo multiple inspections, grades were averaged to create a single "typical" score per restaurant.
    * The dataset was split: **75% training data, 25% test data.**

    After cleaning, **2,768 unique graded restaurants remained**. About 52.6% received an A, while the rest earned lower grades.
    """
)

# -------------------------------------------------
#  Approach
# -------------------------------------------------
st.markdown('<h2 style="font-size: 1.75rem;">Approach</h2>', unsafe_allow_html=True)
st.markdown(
    """
    The approach is to evaluate each feature individually--based solely on information available to a customer--and estimate the **probability** that a restaurant receives an A based on that single feature. These probabilities are then combined using **logistic regression** to produce a final prediction.

    To support effective feature selection:

    * Each feature is tested independently with its own held-out test set.
    * The model output for each feature is simply the **predicted probability of an A.**

    Two important considerations arise:

    1.  Features fall into different categories: **Textual** (name, street address), **Independent categorical** (borough, cuisine), and **Ordered/related numeric** (ZIP code).
    2.  Some features are strongly correlated--e.g., ZIP code to borough, or name to cuisine--which limits which features can be combined without harming performance. **Models avoid using pairs of features that are clearly correlated.**
    """
)

# -------------------------------------------------
#  Feature Analyses
# -------------------------------------------------
st.markdown('<h2 style="font-size: 1.75rem;">Feature Analyses</h2>', unsafe_allow_html=True)

st.subheader("Borough")
st.markdown(
    """
    Using **Bayes' Rule** with indicator functions and evaluating with a Naive Bayes classifier:

    * Staten Island alone achieves **74.2% accuracy**.
    * Citywide accuracy is **51.3%**, effectively random.

    This suggests borough is too coarse--boroughs are large and diverse. Staten Island performs better mainly because it has a more homogeneous population.
    """
)

st.subheader("Food Type")
st.markdown(
    """
    Using Naive Bayes on cuisine type yields **61.4% accuracy**, significantly better than borough.

    * **Strong predictors** (generalization error < 30%): Asian (Chinese/Japanese), Donuts, Greek, Ice Cream, Indian, Indonesian, Juice/Smoothie, Russian, Sandwiches, Steak, Turkish.
    * **Weak predictors** (error > 45%): African, Bagels, Bakery, Italian, Korean, Kosher, Middle Eastern, Pizza, Spanish, Vegetarian.

    Food type is therefore a valuable feature.
    """
)

st.subheader("Name")
st.markdown(
    """
    Restaurant names were evaluated using a **Naive Bayes text classifier** similar to standard spam filters.

    * Accuracy: **58.3%**, statistically significant.
    * Training set produced 2,948 unique words.
    * Words strongly associated with grade predictions include: **Bombay, Chen, Dunkin, Donuts, Fusion, Ice, Pain, Quotidien, Shoppe, Town.**

    Many of these correlate with cuisine type (e.g., "Donuts," "Bombay," "Ice"). Because of this overlap, **name and cuisine should not be combined** in the final model.
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

    1.  **Naive Bayes:** Accuracy: 55.7% across 188 ZIP codes. Statistically meaningful but weaker than name or food type.
    2.  **k-means Clustering:** ZIP codes grouped into 42 clusters (matching NYC's 42 recognized neighborhoods). Accuracy: 53%, slightly worse than Naive Bayes.
    3.  **Polynomial Fitting:**
        * Group ZIP codes into seven natural numeric clusters (e.g., 103xx -> Staten Island).
        * Encode A = 1, else = 0.
        * Use cross-validation to find optimal polynomial degree.
        * Fit a polynomial to each group to estimate probabilities.
        * Accuracy: 53.1%, slightly above random.

    While polynomial fitting captures adjacency patterns, it performs worse than Naive Bayes when used alone.
    """
)

# -------------------------------------------------
#  Combining Features
# -------------------------------------------------
st.markdown('<h2 style="font-size: 1.75rem;">Combining Features</h2>', unsafe_allow_html=True)
st.markdown(
    """
    To improve prediction, the strongest independent features are combined using **logistic regression**:

    * **Inputs:** Probability based on **food type** and Naive Bayes probability from **ZIP code**.
    * Training uses gradient descent with a decreasing learning rate until convergence.
    """
)

# Close newspaper-body DIV
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

col_res, col_why = st.columns(2)

with col_res:
    st.subheader("Final Results")
    st.markdown(
        """
        * Logistic regression with food type + Naive Bayes ZIP yields similar accuracy to food type alone.
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
