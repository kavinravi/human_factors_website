import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="ML Algorithm Explorer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

ALGORITHMS = [
    {
        "name": "Linear Regression",
        "tags": ["supervised", "regression"],
        "image": "linear_regression.png",
        "strengths": "Simple, highly interpretable, very fast to train and predict; coefficients directly show feature importance and direction",
        "weaknesses": "Assumes linear relationships; sensitive to outliers; struggles with multicollinearity; cannot capture complex patterns without manual feature engineering"
    },
    {
        "name": "Decision Trees (Regression)",
        "tags": ["supervised", "regression"],
        "image": "decision_trees.png",
        "strengths": "Highly interpretable (visualizable); handles non-linear relationships naturally; no feature scaling required; handles mixed data types",
        "weaknesses": "Prone to overfitting; unstable (small data changes cause large tree changes); poor extrapolation beyond training data range"
    },
    {
        "name": "Random Forest (Regression)",
        "tags": ["supervised", "regression"],
        "image": "random_forest.png",
        "strengths": "Robust to overfitting compared to single trees; handles non-linearity and interactions well; provides feature importance; works well out-of-the-box with minimal tuning",
        "weaknesses": "Less interpretable than single trees; slower training/prediction than linear models; memory-intensive for large forests; still struggles with extrapolation"
    },
    {
        "name": "Gradient Boosting (Regression)",
        "tags": ["supervised", "regression"],
        "image": "gradient_boosting_forest.png",
        "subtitle": "XGBoost, LightGBM, CatBoost",
        "strengths": "Often achieves state-of-the-art accuracy on tabular data; handles non-linearity and interactions; can handle missing values (implementation-dependent); feature importance available",
        "weaknesses": "Requires careful hyperparameter tuning; prone to overfitting without regularization; slower to train than random forests; less interpretable"
    },
    {
        "name": "K-Nearest Neighbors (Regression)",
        "tags": ["supervised", "regression"],
        "image": "KNN.png",
        "strengths": "Simple and intuitive; no training phase (lazy learner); naturally captures non-linear relationships; no assumptions about data distribution",
        "weaknesses": "Slow at prediction time for large datasets (must search all points); sensitive to feature scaling; suffers in high dimensions (curse of dimensionality); requires meaningful distance metric; no model to interpret"
    },
    {
        "name": "Logistic Regression",
        "tags": ["supervised", "classification"],
        "image": "logistic_regression.png",
        "strengths": "Simple, fast, and highly interpretable; outputs well-calibrated probabilities; works well with linearly separable data; easily extends to multiclass (one-vs-rest, multinomial)",
        "weaknesses": "Assumes linear decision boundary; struggles with complex non-linear patterns; sensitive to outliers and multicollinearity"
    },
    {
        "name": "Decision Trees (Classification)",
        "tags": ["supervised", "classification"],
        "image": "decision_trees.png",
        "strengths": "Highly interpretable; handles non-linear boundaries naturally; no feature scaling needed; outputs class probabilities; handles multiclass natively",
        "weaknesses": "Prone to overfitting; high variance (unstable); can create biased splits with imbalanced classes"
    },
    {
        "name": "Random Forest (Classification)",
        "tags": ["supervised", "classification"],
        "image": "random_forest.png",
        "strengths": "Robust to overfitting; handles high-dimensional data well; provides feature importance and OOB error estimates; works well with imbalanced data (with class weighting)",
        "weaknesses": "Less interpretable than single trees; can be slow for real-time prediction with many trees; biased toward features with many categories"
    },
    {
        "name": "Gradient Boosting (Classification)",
        "tags": ["supervised", "classification"],
        "image": "gradient_boosting_forest.png",
        "subtitle": "XGBoost, LightGBM, CatBoost",
        "strengths": "Often best-in-class accuracy on tabular data; handles class imbalance well; built-in regularization options; can handle missing values",
        "weaknesses": "Requires careful tuning to avoid overfitting; computationally expensive to train; less interpretable; sensitive to noisy labels"
    },
    {
        "name": "Support Vector Machines (SVM)",
        "tags": ["supervised", "classification"],
        "image": "SVM.png",
        "strengths": "Effective in high-dimensional spaces; works well with clear margin of separation; kernel trick enables non-linear boundaries; memory-efficient (uses support vectors only)",
        "weaknesses": "Computationally expensive for large datasets (O(n²) to O(n³)); sensitive to feature scaling; requires kernel selection and tuning; probability estimates require additional computation (Platt scaling); binary by nature (multiclass requires one-vs-one or one-vs-rest)"
    },
    {
        "name": "K-Nearest Neighbors (Classification)",
        "tags": ["supervised", "classification"],
        "image": "KNN.png",
        "strengths": "Simple and intuitive; no training phase (lazy learner); naturally handles multiclass; decision boundaries can be arbitrarily complex; no assumptions about data distribution",
        "weaknesses": "Slow at prediction time for large datasets (must search all points); sensitive to feature scaling and irrelevant features; suffers in high dimensions (curse of dimensionality); requires meaningful distance metric; no model to interpret"
    },
    {
        "name": "Naive Bayes",
        "tags": ["supervised", "classification"],
        "image": "naive_bayes.png",
        "strengths": "Extremely fast to train and predict; works well with high-dimensional data (e.g., text); handles multiclass natively; performs surprisingly well despite strong independence assumption; good baseline for text classification",
        "weaknesses": "Assumes feature independence (rarely true in practice); poor probability estimates (often overconfident); sensitive to feature representation; struggles when features are correlated"
    },
    {
        "name": "K-Means",
        "tags": ["unsupervised", "clustering"],
        "image": "kmeans.png",
        "strengths": "Simple and fast (O(n)); scales well to large datasets; works well with spherical, evenly-sized clusters; easy to interpret",
        "weaknesses": "Must specify K in advance; assumes spherical clusters of similar size; sensitive to initialization and outliers; only finds convex clusters"
    },
    {
        "name": "Gaussian Mixture Models (GMM)",
        "tags": ["unsupervised", "clustering"],
        "image": "gaussian_mixture_models.png",
        "strengths": "Soft clustering (probabilistic assignments); can model elliptical clusters of different shapes/sizes; provides density estimation; handles overlapping clusters",
        "weaknesses": "Must specify number of components; sensitive to initialization; assumes Gaussian distributions; computationally more expensive than K-Means; can converge to local optima"
    },
    {
        "name": "DBSCAN",
        "tags": ["unsupervised", "clustering"],
        "image": "DBSCAN.png",
        "strengths": "No need to specify number of clusters; finds arbitrarily shaped clusters; robust to outliers (marks them as noise); only two hyperparameters (eps, min_samples)",
        "weaknesses": "Struggles with varying density clusters; sensitive to hyperparameter choices; not suitable for high-dimensional data without preprocessing; no soft assignments"
    },
    {
        "name": "Spectral Clustering",
        "tags": ["unsupervised", "clustering"],
        "image": "spectral_clustering.png",
        "strengths": "Finds non-convex, complex-shaped clusters; based on graph connectivity rather than distance; works well when cluster structure is defined by connectivity",
        "weaknesses": "Computationally expensive (O(n³) for eigendecomposition) — not suitable for large datasets; must specify number of clusters; sensitive to similarity graph construction"
    },
    {
        "name": "Hierarchical Agglomerative Clustering",
        "tags": ["unsupervised", "clustering"],
        "image": "hierarchical_agglomerative.png",
        "strengths": "Produces interpretable dendrogram; no need to pre-specify K (can cut at any level); can use various linkage methods (single, complete, average, Ward)",
        "weaknesses": "Computationally expensive (O(n²) memory, O(n³) time for naive implementation); no reassignment after merging (greedy); sensitive to noise and outliers; doesn't scale well"
    },
    {
        "name": "Principal Component Analysis (PCA)",
        "tags": ["unsupervised", "dimensionality reduction"],
        "image": "PCA.png",
        "strengths": "Fast and deterministic; preserves global variance structure; components are orthogonal and ranked by importance; useful for noise reduction and preprocessing",
        "weaknesses": "Only captures linear relationships; sensitive to feature scaling; components can be hard to interpret; may lose important local structure"
    },
    {
        "name": "T-SNE",
        "tags": ["unsupervised", "dimensionality reduction"],
        "image": "t-sne.png",
        "subtitle": "t-Distributed Stochastic Neighbor Embedding",
        "strengths": "Excellent for 2D/3D visualization; preserves local neighborhood structure; reveals clusters and patterns invisible to PCA",
        "weaknesses": "Computationally expensive (O(n²)); non-deterministic (results vary across runs); perplexity hyperparameter sensitive; distances between clusters are not meaningful; not suitable for dimensionality reduction beyond visualization; cannot transform new data points"
    },
    {
        "name": "UMAP",
        "tags": ["unsupervised", "dimensionality reduction"],
        "image": "UMAP.png",
        "subtitle": "Uniform Manifold Approximation and Projection",
        "strengths": "Faster than T-SNE; better preserves global structure while maintaining local structure; scales to larger datasets; can be used for general dimensionality reduction (not just visualization); supports transforming new data points",
        "weaknesses": "Hyperparameter sensitive (n_neighbors, min_dist); non-deterministic; less theoretical foundation than PCA; can distort densities; results require careful interpretation"
    },
]

ALL_TAGS = ["supervised", "unsupervised", "regression", "classification", "clustering", "dimensionality reduction"]

TAG_COLORS = {
    "supervised": "#10b981",
    "unsupervised": "#8b5cf6",
    "regression": "#3b82f6",
    "classification": "#f59e0b",
    "clustering": "#ec4899",
    "dimensionality reduction": "#06b6d4"
}

def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --bg-primary: #0f0f14;
        --bg-secondary: #1a1a22;
        --bg-card: #1e1e28;
        --bg-card-hover: #26263a;
        --text-primary: #ffffff;
        --text-secondary: #c8c8d4;
        --text-muted: #9898a8;
        --accent-primary: #6366f1;
        --accent-secondary: #818cf8;
        --border-color: #3a3a4a;
        --success-color: #34d399;
        --warning-color: #fbbf24;
    }
    
    .stApp {
        background: linear-gradient(180deg, #0f0f14 0%, #0a0a10 100%) !important;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hide fullscreen button on images */
    button[title="View fullscreen"],
    div[data-testid="StyledFullScreenButton"] {
        display: none !important;
    }
    
    /* Force dark theme colors on all text */
    .stApp, .stApp * {
        color: #ffffff;
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: #e0e0e8 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Main container - minimal top padding */
    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Remove Streamlit's default top spacing */
    .stApp > header,
    [data-testid="stHeader"] {
        display: none !important;
        height: 0 !important;
    }
    
    /* Target Streamlit's app view container */
    [data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
    }
    
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 1rem !important;
    }
    
    /* Reduce gaps between Streamlit elements */
    .stMarkdown {
        margin-bottom: 0 !important;
    }
    
    .element-container {
        margin-bottom: 0 !important;
    }
    
    div[data-testid="stVerticalBlock"] > div {
        gap: 0 !important;
    }
    
    /* Reduce padding in main area */
    section[data-testid="stMain"] > div {
        padding-top: 0 !important;
    }
    
    /* Reduce column gaps */
    div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem !important;
    }
    
    /* Tighten Streamlit element spacing */
    div[data-testid="stElementContainer"] {
        margin-bottom: 0 !important;
    }
    
    /* Force tight spacing on all vertical blocks */
    .stVerticalBlock, [data-testid="stVerticalBlock"] {
        gap: 0.25rem !important;
    }
    
    /* Site header with title - compact */
    .site-header {
        text-align: center;
        padding: 0.5rem 0 0.75rem 0;
        margin-bottom: 0;
    }
    
    .site-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .site-tagline {
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        color: #b8b8c8;
        margin-top: 0.25rem;
    }
    
    /* Navigation wrapper - compact */
    .nav-wrapper {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
        padding: 0.5rem 0 !important;
        margin-bottom: 0.75rem !important;
        border-bottom: none !important;
    }
    
    /* Style segmented control / pills navigation */
    div[data-testid="stSegmentedControl"] {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    
    /* Target the parent container to ensure centering */
    div[data-testid="stSegmentedControl"] > div[role="radiogroup"],
    div[role="radiogroup"] {
        margin: 0 auto !important;
        display: inline-flex !important;
        background: #1e1e28 !important;
        border: 1px solid #3a3a4a !important;
        border-radius: 10px !important;
        padding: 4px !important;
    }
    
    div[data-testid="stSegmentedControl"] button,
    div[role="radiogroup"] button {
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        border-radius: 7px !important;
        padding: 12px 28px !important;
        color: #c8c8d4 !important;
        background: transparent !important;
        border: none !important;
    }
    
    div[data-testid="stSegmentedControl"] button:hover,
    div[role="radiogroup"] button:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[role="radiogroup"] button[aria-pressed="true"] {
        color: #ffffff !important;
        background: #6366f1 !important;
    }
    
    /* Force the container that holds the segmented control to be full width and centered */
    .stElementContainer:has(div[data-testid="stSegmentedControl"]),
    .element-container:has(div[role="radiogroup"]) {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    
    /* Fallback button styling */
    .stButton > button {
        font-family: 'Outfit', sans-serif !important;
        border-radius: 8px !important;
    }
    
    /* Filter dropdown styling */
    .filter-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .filter-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Style the multiselect */
    div[data-baseweb="select"] {
        font-family: 'Outfit', sans-serif;
    }
    
    div[data-baseweb="select"] > div {
        background-color: var(--bg-secondary) !important;
        border-color: var(--border-color) !important;
        font-size: 0.9rem !important;
    }
    
    /* Multiselect container - ensure full width for options */
    div[data-testid="stMultiSelect"] {
        max-width: 350px !important;
    }
    
    /* Dropdown menu options - prevent text truncation */
    div[data-baseweb="popover"] {
        min-width: 280px !important;
    }
    
    div[data-baseweb="popover"] li,
    div[data-baseweb="menu"] li {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
        padding: 10px 16px !important;
        font-size: 0.9rem !important;
    }
    
    /* Selected tag pills in multiselect */
    div[data-baseweb="tag"] {
        background-color: var(--accent-primary) !important;
        border-radius: 6px !important;
        margin: 2px !important;
    }
    
    div[data-baseweb="tag"] span {
        color: white !important;
        font-size: 0.85rem !important;
    }
    
    /* Checkbox styling in multiselect dropdown */
    div[data-baseweb="checkbox"] {
        margin-right: 8px !important;
    }
    
    /* Algorithm cards */
    .algo-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.25rem;
        transition: all 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .algo-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--accent-primary);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
    }
    
    .algo-card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .algo-card-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-bottom: 0.75rem;
    }
    
    .algo-card-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .algo-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-weight: 500;
        text-transform: lowercase;
    }
    
    .algo-card-section {
        margin-bottom: 0.75rem;
    }
    
    .algo-card-section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.35rem;
    }
    
    .strengths-title {
        color: var(--success-color);
    }
    
    .weaknesses-title {
        color: var(--warning-color);
    }
    
    .algo-card-text {
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem;
        color: #d0d0dc;
        line-height: 1.5;
    }
    
    /* Page content header - compact */
    .page-header {
        margin-bottom: 0.75rem;
    }
    
    .page-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.15rem;
    }
    
    .page-subtitle {
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        color: #b0b0c0;
    }
    
    /* Coming soon page */
    .coming-soon {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 50vh;
        text-align: center;
    }
    
    .coming-soon-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
    }
    
    .coming-soon-text {
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        color: #c0c0d0;
        max-width: 500px;
        line-height: 1.6;
    }
    
    /* Streamlit overrides */
    .stButton > button {
        font-family: 'Outfit', sans-serif;
    }
    
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
    
    /* Results count */
    .results-count {
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
    }
    
    .results-count span {
        color: var(--accent-secondary);
        font-weight: 600;
    }
    
    /* Divider - compact */
    .subtle-divider {
        border: none;
        border-top: 1px solid var(--border-color);
        margin: 0.5rem 0;
    }
    
    /* Survey radio button styling - full width, left aligned */
    div[data-testid="stRadio"] {
        width: 100% !important;
    }
    
    div[data-testid="stRadio"] > div {
        gap: 0.75rem !important;
        width: 100% !important;
        flex-direction: column !important;
        align-items: stretch !important;
    }
    
    div[data-testid="stRadio"] > div > label {
        width: 100% !important;
        max-width: 100% !important;
        background: #1e1e28 !important;
        border: 1px solid #3a3a4a !important;
        border-radius: 10px !important;
        padding: 1rem 1.25rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
    }
    
    div[data-testid="stRadio"] > div > label:hover {
        border-color: #6366f1 !important;
        background: #26263a !important;
    }
    
    div[data-testid="stRadio"] > div > label[data-checked="true"],
    div[data-testid="stRadio"] > div > label:has(input:checked) {
        border-color: #6366f1 !important;
        background: rgba(99, 102, 241, 0.15) !important;
    }
    
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label p {
        color: #e0e0e8 !important;
        font-size: 0.95rem !important;
    }
    
    /* Progress bar styling - taller and more visible */
    div[data-testid="stProgress"] {
        margin-bottom: 0.5rem !important;
    }
    
    div[data-testid="stProgress"] > div {
        background-color: #2a2a3a !important;
        height: 8px !important;
        border-radius: 4px !important;
    }
    
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #6366f1, #818cf8) !important;
        height: 8px !important;
        border-radius: 4px !important;
    }
    
    /* Survey button styling - ensure text visibility */
    .stButton > button {
        color: #ffffff !important;
    }
    
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]) {
        background-color: #2a2a3a !important;
        border: 1px solid #4a4a5a !important;
        color: #ffffff !important;
    }
    
    .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover {
        background-color: #3a3a4a !important;
        border-color: #6366f1 !important;
    }
    
    .stButton > button p {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)


def get_tag_style(tag):
    color = TAG_COLORS.get(tag, "#6366f1")
    return f"background: {color}22; color: {color}; border: 1px solid {color}44;"


def render_algorithm_card(algo, assets_path):
    image_path = assets_path / algo["image"]
    
    with st.container():
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        
        st.markdown(f'<div class="algo-card-title">{algo["name"]}</div>', unsafe_allow_html=True)
        
        if "subtitle" in algo:
            st.markdown(f'<div class="algo-card-subtitle">{algo["subtitle"]}</div>', unsafe_allow_html=True)
        
        tags_html = ""
        for tag in algo["tags"]:
            style = get_tag_style(tag)
            tags_html += f'<span class="algo-tag" style="{style}">{tag}</span>'
        st.markdown(f'<div class="algo-card-tags">{tags_html}</div>', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="algo-card-section">
            <div class="algo-card-section-title strengths-title">Strengths</div>
            <div class="algo-card-text">{algo["strengths"]}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="algo-card-section">
            <div class="algo-card-section-title weaknesses-title">Weaknesses</div>
            <div class="algo-card-text">{algo["weaknesses"]}</div>
        </div>
        ''', unsafe_allow_html=True)


def render_site_header():
    st.markdown('''
    <div class="site-header">
        <h1 class="site-title">ML Algorithm Explorer</h1>
        <p class="site-tagline">Find the right machine learning algorithm for your problem</p>
    </div>
    ''', unsafe_allow_html=True)


def render_navigation():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "gallery"
    
    st.markdown('<div class="nav-wrapper">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            page = st.segmented_control(
                "Navigation",
                ["Gallery", "Find My ML Model"],
                default="Gallery" if st.session_state.current_page == "gallery" else "Find My ML Model",
                label_visibility="collapsed"
            )
            if page == "Gallery" and st.session_state.current_page != "gallery":
                st.session_state.current_page = "gallery"
                st.rerun()
            elif page == "Find My ML Model" and st.session_state.current_page != "finder":
                st.session_state.current_page = "finder"
                st.rerun()
        except AttributeError:
            nav_cols = st.columns(2, gap="small")
            with nav_cols[0]:
                if st.button("Gallery", use_container_width=True, 
                            type="primary" if st.session_state.current_page == "gallery" else "secondary"):
                    st.session_state.current_page = "gallery"
                    st.rerun()
            with nav_cols[1]:
                if st.button("Find My ML Model", use_container_width=True,
                            type="primary" if st.session_state.current_page == "finder" else "secondary"):
                    st.session_state.current_page = "finder"
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def gallery_page():
    assets_path = Path(__file__).parent / "assets"
    
    st.markdown('''
    <div class="page-header">
        <div class="page-title">Algorithm Gallery</div>
        <div class="page-subtitle">Explore machine learning algorithms and their trade-offs</div>
    </div>
    ''', unsafe_allow_html=True)
    
    if "selected_tags" not in st.session_state:
        st.session_state.selected_tags = []
    
    tag_options = {
        "supervised": "Supervised",
        "unsupervised": "Unsupervised", 
        "regression": "Regression",
        "classification": "Classification",
        "clustering": "Clustering",
        "dimensionality reduction": "Dimensionality Reduction"
    }
    
    st.markdown('<p style="font-size: 0.85rem; color: #9898a8; margin-bottom: 0.5rem;">Filter by tag (select multiple):</p>', unsafe_allow_html=True)
    
    selected = st.multiselect(
        "Filter by tags",
        options=list(tag_options.keys()),
        default=st.session_state.selected_tags,
        format_func=lambda x: tag_options[x],
        label_visibility="collapsed",
        placeholder="All algorithms (click to filter)"
    )
    
    if selected != st.session_state.selected_tags:
        st.session_state.selected_tags = selected
        st.rerun()
    
    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
    
    if not st.session_state.selected_tags:
        filtered_algos = ALGORITHMS
    else:
        filtered_algos = [
            algo for algo in ALGORITHMS
            if all(tag in algo["tags"] for tag in st.session_state.selected_tags)
        ]
    
    st.markdown(f'<div class="results-count">Showing <span>{len(filtered_algos)}</span> of {len(ALGORITHMS)} algorithms</div>', unsafe_allow_html=True)
    
    cols_per_row = 3
    for i in range(0, len(filtered_algos), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(filtered_algos):
                with col:
                    render_algorithm_card(filtered_algos[i + j], assets_path)
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)


SURVEY_QUESTIONS = {
    "Q1": {
        "text": "Which of the following best describes your problem?",
        "options": [
            ("supervised", "I have input features and a known outcome (label) I want to predict."),
            ("clustering", "I want to discover natural groups or segments in my data."),
            ("dimreduction", "I want to reduce the number of features or visualize high-dimensional data.")
        ]
    },
    "Q2": {
        "text": "What best describes your target (output) variable?",
        "options": [
            ("regression", "A numeric value (e.g., price, temperature, time)."),
            ("binary", "A binary category (two classes, e.g., spam vs not spam)."),
            ("multiclass", "A categorical label with more than two classes (e.g., dog/cat/bird).")
        ]
    },
    "Q3": {
        "text": "Approximately how many data points (rows) does your dataset have?",
        "options": [
            (1, "Fewer than 1,000"),
            (2, "1,000 - 10,000"),
            (3, "10,000 - 100,000"),
            (4, "100,000 - 1,000,000"),
            (5, "More than 1,000,000")
        ]
    },
    "Q4": {
        "text": "Roughly how many features (columns) does your dataset have?",
        "options": [
            (1, "Fewer than 10"),
            (2, "10 - 100"),
            (3, "100 - 1,000"),
            (4, "More than 1,000")
        ]
    },
    "Q5": {
        "text": "Do you need very fast predictions in real-time?",
        "options": [
            (True, "Yes, prediction speed is critical."),
            (False, "No, a small delay per prediction is acceptable.")
        ]
    },
    "Q6": {
        "text": "Is your input data primarily text (e.g., emails, reviews, documents)?",
        "options": [
            (True, "Yes, the inputs are primarily text."),
            (False, "No, the inputs are mainly numeric/categorical features.")
        ]
    },
    "Q7": {
        "text": "Which is more important for your use case?",
        "options": [
            (1, "High interpretability (easy to explain to non-experts)."),
            (2, "A balance between interpretability and accuracy."),
            (3, "Highest possible predictive performance, even if the model is a black box.")
        ]
    },
    "Q8": {
        "text": "Do you expect many outliers or noisy points in your data?",
        "options": [
            (True, "Yes, there are many outliers/noisy points."),
            (False, "No, the data is mostly clean with few outliers.")
        ]
    },
    "Q9": {
        "text": "Are your classes highly imbalanced (one class is much rarer)?",
        "options": [
            (True, "Yes, the classes are highly imbalanced."),
            (False, "No, the classes are reasonably balanced.")
        ]
    },
    "Q12": {
        "text": "Do you have a reasonable guess for the number of clusters/groups you expect?",
        "options": [
            (True, "Yes, I have a rough idea of how many clusters there are."),
            (False, "No, I do not know how many clusters there are.")
        ]
    },
    "Q13": {
        "text": "Is explicitly identifying outliers/noise points important for your task?",
        "options": [
            (True, "Yes, I want to detect and separate outliers/noise."),
            (False, "No, I mainly care about the main clusters.")
        ]
    },
    "Q14": {
        "text": "Which description best matches the clusters you expect?",
        "options": [
            (1, "Roughly spherical or compact blob-like clusters of similar size."),
            (2, "Arbitrarily shaped clusters (e.g., chains, rings) or clusters of very different densities."),
            (3, "I am not sure.")
        ]
    },
    "Q15": {
        "text": "Do you want a probability for each point belonging to each cluster (soft assignments)?",
        "options": [
            (True, "Yes, I want probabilistic cluster memberships."),
            (False, "No, a single cluster label per point is enough.")
        ]
    },
    "Q16": {
        "text": "Is it important to see a hierarchy of clusters (e.g., via a tree/dendrogram)?",
        "options": [
            (True, "Yes, I want to see clusters at multiple levels of granularity."),
            (False, "No, a single flat partition is enough.")
        ]
    },
    "Q17": {
        "text": "Is your dataset small enough that slower algorithms are acceptable?",
        "options": [
            (True, "Yes, my dataset is small (fewer than ~10,000 points)."),
            (False, "No, my dataset is larger, and I need methods that scale well.")
        ]
    },
    "Q18": {
        "text": "Are you okay with some data points being left unassigned (marked as noise)?",
        "options": [
            (True, "Yes, I am okay with some points being left unclassified."),
            (False, "No, I want every point assigned to some cluster.")
        ]
    },
    "Q19": {
        "text": "What is the main reason you want to reduce dimensionality?",
        "options": [
            (1, "General preprocessing / noise reduction before other models."),
            (2, "2D/3D visualization of the data to see clusters/patterns."),
            (3, "Both preprocessing and visualization are important.")
        ]
    },
    "Q20": {
        "text": "Will you need to apply the same dimensionality reduction to new data points later?",
        "options": [
            (True, "Yes, I will need to transform new points consistently."),
            (False, "No, reducing a fixed dataset once is enough.")
        ]
    },
    "Q21": {
        "text": "What do you care about more in the reduced space?",
        "options": [
            (1, "Preserving overall global variance/structure."),
            (2, "Preserving small-scale local neighborhoods (local clusters)."),
            (3, "A balance between global structure and local neighborhoods.")
        ]
    },
    "Q22": {
        "text": "Is your dataset small or medium-sized so that slower visualization methods are acceptable?",
        "options": [
            (True, "Yes, my dataset is small/medium (fewer than ~50,000 points)."),
            (False, "No, my dataset is large and I need faster methods.")
        ]
    },
    "Q23": {
        "text": "Is it important that you get the exact same result every time you run?",
        "options": [
            (True, "Yes, I need deterministic results."),
            (False, "No, slight randomness between runs is acceptable.")
        ]
    }
}

MODEL_INFO = {
    "Linear Regression": {"category": "regression", "tags": ["supervised", "regression"]},
    "Decision Trees (Regression)": {"category": "regression", "tags": ["supervised", "regression"]},
    "Random Forest (Regression)": {"category": "regression", "tags": ["supervised", "regression"]},
    "Gradient Boosting (Regression)": {"category": "regression", "tags": ["supervised", "regression"]},
    "K-Nearest Neighbors (Regression)": {"category": "regression", "tags": ["supervised", "regression"]},
    "Logistic Regression": {"category": "classification", "tags": ["supervised", "classification"]},
    "Decision Trees (Classification)": {"category": "classification", "tags": ["supervised", "classification"]},
    "Random Forest (Classification)": {"category": "classification", "tags": ["supervised", "classification"]},
    "Gradient Boosting (Classification)": {"category": "classification", "tags": ["supervised", "classification"]},
    "Support Vector Machines (SVM)": {"category": "classification", "tags": ["supervised", "classification"]},
    "K-Nearest Neighbors (Classification)": {"category": "classification", "tags": ["supervised", "classification"]},
    "Naive Bayes": {"category": "classification", "tags": ["supervised", "classification"]},
    "K-Means": {"category": "clustering", "tags": ["unsupervised", "clustering"]},
    "Gaussian Mixture Models (GMM)": {"category": "clustering", "tags": ["unsupervised", "clustering"]},
    "DBSCAN": {"category": "clustering", "tags": ["unsupervised", "clustering"]},
    "Spectral Clustering": {"category": "clustering", "tags": ["unsupervised", "clustering"]},
    "Hierarchical Agglomerative Clustering": {"category": "clustering", "tags": ["unsupervised", "clustering"]},
    "Principal Component Analysis (PCA)": {"category": "dimreduction", "tags": ["unsupervised", "dimensionality reduction"]},
    "T-SNE": {"category": "dimreduction", "tags": ["unsupervised", "dimensionality reduction"]},
    "UMAP": {"category": "dimreduction", "tags": ["unsupervised", "dimensionality reduction"]}
}

def get_branch_questions(branch, answers):
    if branch == "supervised":
        target_type = answers.get("Q2")
        if target_type == "regression":
            return ["Q2", "Q3", "Q4", "Q5", "Q7", "Q8"]
        else:
            return ["Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"]
    elif branch == "clustering":
        # Q17 removed - we infer from Q3 (dataset size already captured)
        return ["Q3", "Q4", "Q7", "Q8", "Q12", "Q13", "Q14", "Q15", "Q16", "Q18"]
    else:
        # Q22 removed - we infer from Q3 (dataset size already captured)
        return ["Q3", "Q4", "Q7", "Q19", "Q20", "Q21", "Q23"]

def infer_derived_answers(answers):
    """Infer answers to redundant questions from Q3 (dataset size)"""
    q3 = answers.get("Q3", 2)
    # Q17: Is dataset small (<10k)? Infer from Q3
    # Q3: 1=<1k, 2=1k-10k, 3=10k-100k, 4=100k-1M, 5=>1M
    answers["Q17"] = q3 <= 2  # True if ≤10k rows
    # Q22: Is dataset small/medium (<50k)? Infer from Q3
    answers["Q22"] = q3 <= 3  # True if ≤100k rows (approximation for <50k)
    return answers

def score_regression_models(answers):
    scores = {}
    q3 = answers.get("Q3", 2)
    q4 = answers.get("Q4", 2)
    q5 = answers.get("Q5", False)
    q7 = answers.get("Q7", 2)
    q8 = answers.get("Q8", False)
    
    scores["Linear Regression"] = 50
    if q7 == 1: scores["Linear Regression"] += 30
    if q3 >= 2: scores["Linear Regression"] += 10
    if not q8: scores["Linear Regression"] += 15
    if q5: scores["Linear Regression"] += 10
    
    scores["Decision Trees (Regression)"] = 50
    if q7 in [1, 2]: scores["Decision Trees (Regression)"] += 20
    if q8: scores["Decision Trees (Regression)"] += 20
    
    scores["Random Forest (Regression)"] = 50
    if q3 >= 2: scores["Random Forest (Regression)"] += 20
    else: scores["Random Forest (Regression)"] -= 20
    if q7 in [2, 3]: scores["Random Forest (Regression)"] += 20
    if q8: scores["Random Forest (Regression)"] += 15
    if not q5: scores["Random Forest (Regression)"] += 10
    
    scores["Gradient Boosting (Regression)"] = 50
    if q3 >= 2: scores["Gradient Boosting (Regression)"] += 20
    else: scores["Gradient Boosting (Regression)"] -= 20
    if q7 == 3: scores["Gradient Boosting (Regression)"] += 25
    if not q5: scores["Gradient Boosting (Regression)"] += 10
    if not q8: scores["Gradient Boosting (Regression)"] += 5
    
    scores["K-Nearest Neighbors (Regression)"] = 50
    if q3 <= 2: scores["K-Nearest Neighbors (Regression)"] += 20
    else: scores["K-Nearest Neighbors (Regression)"] -= 40
    if q4 <= 2: scores["K-Nearest Neighbors (Regression)"] += 15
    else: scores["K-Nearest Neighbors (Regression)"] -= 30
    if q7 == 2: scores["K-Nearest Neighbors (Regression)"] += 10
    if not q5: scores["K-Nearest Neighbors (Regression)"] += 10
    
    return scores

def score_classification_models(answers):
    scores = {}
    q3 = answers.get("Q3", 2)
    q4 = answers.get("Q4", 2)
    q5 = answers.get("Q5", False)
    q6 = answers.get("Q6", False)
    q7 = answers.get("Q7", 2)
    q8 = answers.get("Q8", False)
    q9 = answers.get("Q9", False)
    
    scores["Logistic Regression"] = 50
    if q6: scores["Logistic Regression"] += 20
    if q7 == 1: scores["Logistic Regression"] += 25
    if q5: scores["Logistic Regression"] += 15
    if q9: scores["Logistic Regression"] += 10
    
    scores["Decision Trees (Classification)"] = 50
    if q7 in [1, 2]: scores["Decision Trees (Classification)"] += 20
    if q8: scores["Decision Trees (Classification)"] += 20
    
    scores["Random Forest (Classification)"] = 50
    if q3 >= 2: scores["Random Forest (Classification)"] += 20
    else: scores["Random Forest (Classification)"] -= 15
    if q7 in [2, 3]: scores["Random Forest (Classification)"] += 20
    if q8: scores["Random Forest (Classification)"] += 15
    if q9: scores["Random Forest (Classification)"] += 15
    
    scores["Gradient Boosting (Classification)"] = 50
    if q3 >= 2: scores["Gradient Boosting (Classification)"] += 20
    else: scores["Gradient Boosting (Classification)"] -= 15
    if q7 == 3: scores["Gradient Boosting (Classification)"] += 25
    if q9: scores["Gradient Boosting (Classification)"] += 15
    if not q5: scores["Gradient Boosting (Classification)"] += 10
    
    scores["Support Vector Machines (SVM)"] = 50
    if q3 <= 3: scores["Support Vector Machines (SVM)"] += 20
    else: scores["Support Vector Machines (SVM)"] -= 50
    if q7 == 3: scores["Support Vector Machines (SVM)"] += 20
    if q4 <= 2: scores["Support Vector Machines (SVM)"] += 10
    if not q5: scores["Support Vector Machines (SVM)"] += 10
    
    scores["K-Nearest Neighbors (Classification)"] = 50
    if q3 <= 2: scores["K-Nearest Neighbors (Classification)"] += 20
    else: scores["K-Nearest Neighbors (Classification)"] -= 40
    if q4 <= 2: scores["K-Nearest Neighbors (Classification)"] += 15
    else: scores["K-Nearest Neighbors (Classification)"] -= 25
    if q7 == 2: scores["K-Nearest Neighbors (Classification)"] += 10
    if not q5: scores["K-Nearest Neighbors (Classification)"] += 10
    
    scores["Naive Bayes"] = 50
    if q6: scores["Naive Bayes"] += 35
    if q7 in [1, 2]: scores["Naive Bayes"] += 15
    if q5: scores["Naive Bayes"] += 20
    
    return scores

def score_clustering_models(answers):
    scores = {}
    q3 = answers.get("Q3", 2)
    q4 = answers.get("Q4", 2)
    q7 = answers.get("Q7", 2)
    q8 = answers.get("Q8", False)
    q12 = answers.get("Q12", True)
    q13 = answers.get("Q13", False)
    q14 = answers.get("Q14", 1)
    q15 = answers.get("Q15", False)
    q16 = answers.get("Q16", False)
    q17 = answers.get("Q17", True)
    q18 = answers.get("Q18", True)
    
    scores["K-Means"] = 50
    if q12: scores["K-Means"] += 20
    if q14 in [1, 3]: scores["K-Means"] += 15
    if q7 in [1, 2]: scores["K-Means"] += 10
    if not q13: scores["K-Means"] += 10
    if q8: scores["K-Means"] -= 15
    
    scores["Gaussian Mixture Models (GMM)"] = 50
    if q3 <= 4: scores["Gaussian Mixture Models (GMM)"] += 15
    else: scores["Gaussian Mixture Models (GMM)"] -= 20
    if q12: scores["Gaussian Mixture Models (GMM)"] += 15
    if q14 in [1, 3]: scores["Gaussian Mixture Models (GMM)"] += 10
    if q15: scores["Gaussian Mixture Models (GMM)"] += 25
    
    scores["DBSCAN"] = 50
    if q3 <= 3 and q4 <= 2: scores["DBSCAN"] += 20
    else: scores["DBSCAN"] -= 15
    if not q18: scores["DBSCAN"] -= 40
    if q13: scores["DBSCAN"] += 25
    if q14 == 2: scores["DBSCAN"] += 20
    if not q12: scores["DBSCAN"] += 15
    
    scores["Spectral Clustering"] = 50
    if q17: scores["Spectral Clustering"] += 25
    else: scores["Spectral Clustering"] -= 40
    if q14 == 2: scores["Spectral Clustering"] += 25
    if q7 == 3: scores["Spectral Clustering"] += 15
    
    scores["Hierarchical Agglomerative Clustering"] = 50
    if q3 <= 3: scores["Hierarchical Agglomerative Clustering"] += 20
    else: scores["Hierarchical Agglomerative Clustering"] -= 30
    if q16: scores["Hierarchical Agglomerative Clustering"] += 30
    if q7 in [1, 2]: scores["Hierarchical Agglomerative Clustering"] += 15
    
    return scores

def score_dimreduction_models(answers):
    scores = {}
    q3 = answers.get("Q3", 2)
    q7 = answers.get("Q7", 2)
    q19 = answers.get("Q19", 1)
    q20 = answers.get("Q20", False)
    q21 = answers.get("Q21", 1)
    q22 = answers.get("Q22", True)
    q23 = answers.get("Q23", False)
    
    scores["Principal Component Analysis (PCA)"] = 50
    if q19 in [1, 3]: scores["Principal Component Analysis (PCA)"] += 20
    if q21 == 1: scores["Principal Component Analysis (PCA)"] += 20
    if q23: scores["Principal Component Analysis (PCA)"] += 25
    if q3 >= 4: scores["Principal Component Analysis (PCA)"] += 15
    
    scores["T-SNE"] = 50
    if q22: scores["T-SNE"] += 25
    else: scores["T-SNE"] -= 40
    if q19 in [2, 3]: scores["T-SNE"] += 20
    if q21 == 2: scores["T-SNE"] += 20
    if not q20: scores["T-SNE"] += 10
    if not q23: scores["T-SNE"] += 5
    
    scores["UMAP"] = 50
    if q19 in [2, 3]: scores["UMAP"] += 20
    if q20: scores["UMAP"] += 20
    if q21 == 3: scores["UMAP"] += 20
    if q22 or q3 <= 4: scores["UMAP"] += 10
    if not q23: scores["UMAP"] += 5
    
    return scores

def get_current_scores(answers):
    """Calculate scores based on current answers (may be partial)"""
    branch = answers.get("Q1")
    if not branch:
        return None, []
    
    answers_with_inferred = infer_derived_answers(answers.copy())
    
    if branch == "supervised":
        target_type = answers_with_inferred.get("Q2", "regression")
        if target_type == "regression":
            scores = score_regression_models(answers_with_inferred)
        else:
            scores = score_classification_models(answers_with_inferred)
    elif branch == "clustering":
        scores = score_clustering_models(answers_with_inferred)
    else:
        scores = score_dimreduction_models(answers_with_inferred)
    
    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return branch, sorted_models[:3]

def render_live_leaderboard(answers):
    branch, top_models = get_current_scores(answers)
    
    if not branch or not top_models:
        return
    
    short_names = {
        "K-Nearest Neighbors (Regression)": "KNN Regression",
        "K-Nearest Neighbors (Classification)": "KNN Classification",
        "Random Forest (Regression)": "Random Forest",
        "Random Forest (Classification)": "Random Forest",
        "Gradient Boosting (Regression)": "Gradient Boosting",
        "Gradient Boosting (Classification)": "Gradient Boosting",
        "Decision Trees (Regression)": "Decision Trees",
        "Decision Trees (Classification)": "Decision Trees",
        "Support Vector Machines (SVM)": "SVM",
        "Principal Component Analysis (PCA)": "PCA",
        "Gaussian Mixture Models (GMM)": "GMM",
        "Hierarchical Agglomerative Clustering": "Hierarchical",
    }
    
    rank_colors = ["#fbbf24", "#a8a8b8", "#cd7f32"]
    
    html_parts = []
    html_parts.append('<div style="background: linear-gradient(135deg, #1a1a24 0%, #14141c 100%); border: 1px solid #3a3a4a; border-radius: 12px; padding: 1rem 1.25rem; margin-top: 0.5rem;">')
    html_parts.append('<p style="font-size: 0.7rem; color: #6366f1; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; font-weight: 600;">Live Rankings</p>')
    
    for i, (model, score) in enumerate(top_models):
        display_name = short_names.get(model, model)
        if len(display_name) > 20:
            display_name = display_name[:17] + "..."
        color = rank_colors[i]
        border = "border-bottom: 1px solid #2a2a3a;" if i < 2 else ""
        html_parts.append(f'<div style="display: flex; align-items: center; gap: 0.6rem; padding: 0.5rem 0; {border}">')
        html_parts.append(f'<div style="width: 26px; height: 26px; background: {color}; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">')
        html_parts.append(f'<span style="color: #0f0f14; font-weight: 700; font-size: 0.75rem;">{i+1}</span>')
        html_parts.append('</div>')
        html_parts.append(f'<span style="color: #e0e0e8; font-size: 0.85rem; font-weight: 500;">{display_name}</span>')
        html_parts.append('</div>')
    
    html_parts.append('</div>')
    
    st.markdown(''.join(html_parts), unsafe_allow_html=True)

def get_model_justification(model, answers, branch):
    justifications = {
        "Linear Regression": "Simple, fast, and highly interpretable with clear coefficient meanings.",
        "Decision Trees (Regression)": "Handles non-linear patterns and outliers well with easy visualization.",
        "Random Forest (Regression)": "Robust ensemble that handles complexity while providing feature importance.",
        "Gradient Boosting (Regression)": "Often achieves best accuracy on tabular data with proper tuning.",
        "K-Nearest Neighbors (Regression)": "Simple and intuitive for smaller datasets with meaningful distances.",
        "Logistic Regression": "Fast, interpretable, and outputs well-calibrated probabilities.",
        "Decision Trees (Classification)": "Highly interpretable with visual decision rules.",
        "Random Forest (Classification)": "Robust to overfitting with built-in feature importance.",
        "Gradient Boosting (Classification)": "State-of-the-art accuracy on tabular classification tasks.",
        "Support Vector Machines (SVM)": "Effective in high-dimensional spaces with kernel flexibility.",
        "K-Nearest Neighbors (Classification)": "Simple, no training phase, naturally handles multiclass.",
        "Naive Bayes": "Extremely fast, works well with text data and high dimensions.",
        "K-Means": "Fast, scalable, and easy to interpret for spherical clusters.",
        "Gaussian Mixture Models (GMM)": "Provides soft probabilistic assignments and handles elliptical clusters.",
        "DBSCAN": "Finds arbitrary shapes, detects outliers, no need to specify K.",
        "Spectral Clustering": "Excellent for complex, non-convex cluster structures.",
        "Hierarchical Agglomerative Clustering": "Produces interpretable dendrograms for multi-level analysis.",
        "Principal Component Analysis (PCA)": "Fast, deterministic, preserves global variance structure.",
        "T-SNE": "Excellent for visualization, reveals local cluster structure.",
        "UMAP": "Fast, preserves both global and local structure, can transform new points."
    }
    return justifications.get(model, "")

def render_survey_question(question_id, question_data):
    st.markdown(f'<p style="font-size: 1.1rem; color: #ffffff; margin-bottom: 1rem; font-weight: 500;">{question_data["text"]}</p>', unsafe_allow_html=True)
    
    options = question_data["options"]
    option_labels = [opt[1] for opt in options]
    option_values = [opt[0] for opt in options]
    
    selected_idx = st.radio(
        f"q_{question_id}",
        range(len(options)),
        format_func=lambda x: option_labels[x],
        label_visibility="collapsed",
        key=f"radio_{question_id}"
    )
    
    return option_values[selected_idx] if selected_idx is not None else None

def finder_page():
    if "survey_step" not in st.session_state:
        st.session_state.survey_step = 0
    if "survey_answers" not in st.session_state:
        st.session_state.survey_answers = {}
    if "survey_complete" not in st.session_state:
        st.session_state.survey_complete = False
    
    st.markdown('''
    <div class="page-header">
        <div class="page-title">Find My ML Model</div>
        <div class="page-subtitle">Answer a few questions to discover the best algorithm for your problem</div>
    </div>
    ''', unsafe_allow_html=True)
    
    if st.session_state.survey_complete:
        render_survey_results()
        return
    
    answers = st.session_state.survey_answers
    step = st.session_state.survey_step
    
    if step == 0:
        q1_data = SURVEY_QUESTIONS["Q1"]
        answer = render_survey_question("Q1", q1_data)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Next", use_container_width=True, type="primary"):
                st.session_state.survey_answers["Q1"] = answer
                st.session_state.survey_step = 1
                st.rerun()
    else:
        branch = answers.get("Q1", "supervised")
        question_order = get_branch_questions(branch, answers)
        current_q_idx = step - 1
        
        if current_q_idx < len(question_order):
            q_id = question_order[current_q_idx]
            q_data = SURVEY_QUESTIONS[q_id]
            
            # Two-column layout: question on left, leaderboard on right
            question_col, leaderboard_col = st.columns([3, 1])
            
            with question_col:
                progress = (step) / (len(question_order) + 1)
                st.progress(progress)
                st.markdown(f'<p style="font-size: 0.85rem; color: #9898a8; margin-bottom: 1.5rem;">Question {step} of {len(question_order)}</p>', unsafe_allow_html=True)
                
                answer = render_survey_question(q_id, q_data)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("Back", use_container_width=True):
                        st.session_state.survey_step -= 1
                        st.rerun()
                with col3:
                    if st.button("Next", use_container_width=True, type="primary"):
                        st.session_state.survey_answers[q_id] = answer
                        if current_q_idx + 1 >= len(question_order):
                            st.session_state.survey_complete = True
                        else:
                            st.session_state.survey_step += 1
                        st.rerun()
            
            with leaderboard_col:
                render_live_leaderboard(answers)
        else:
            st.session_state.survey_complete = True
            st.rerun()

def render_survey_results():
    answers = st.session_state.survey_answers
    answers = infer_derived_answers(answers)  # Infer Q17/Q22 from Q3
    branch = answers.get("Q1", "supervised")
    
    if branch == "supervised":
        target_type = answers.get("Q2", "regression")
        if target_type == "regression":
            scores = score_regression_models(answers)
        else:
            scores = score_classification_models(answers)
    elif branch == "clustering":
        scores = score_clustering_models(answers)
    else:
        scores = score_dimreduction_models(answers)
    
    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_models = sorted_models[:3]
    
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;">', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.5rem; font-weight: 600; color: #ffffff; margin-bottom: 0.5rem;">Your Recommended Models</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.95rem; color: #b0b0c0;">Based on your answers, here are the best algorithms for your problem:</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    cols = st.columns(len(top_models))
    
    for i, (model, score) in enumerate(top_models):
        with cols[i]:
            rank_colors = ["#fbbf24", "#c0c0c0", "#cd7f32"]
            rank_color = rank_colors[i] if i < 3 else "#6366f1"
            
            st.markdown(f'''
            <div style="background: #1e1e28; border: 2px solid {rank_color}; border-radius: 12px; padding: 1.5rem; text-align: center; height: 100%;">
                <div style="font-size: 1.5rem; font-weight: 700; color: {rank_color}; margin-bottom: 0.5rem;">#{i+1}</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #ffffff; margin-bottom: 0.75rem;">{model}</div>
                <div style="font-size: 0.85rem; color: #b0b0c0; line-height: 1.5;">{get_model_justification(model, answers, branch)}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start Over", use_container_width=True):
            st.session_state.survey_step = 0
            st.session_state.survey_answers = {}
            st.session_state.survey_complete = False
            st.rerun()


def main():
    load_css()
    render_site_header()
    render_navigation()
    
    if st.session_state.current_page == "gallery":
        gallery_page()
    else:
        finder_page()


if __name__ == "__main__":
    main()
