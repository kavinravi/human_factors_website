import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ML Algorithm Explorer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Algorithm data with tags and image mappings
ALGORITHMS = [
    # Supervised - Regression
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
    # Supervised - Classification
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
    # Unsupervised - Clustering
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
    # Dimensionality Reduction
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

# All available tags
ALL_TAGS = ["supervised", "unsupervised", "regression", "classification", "clustering", "dimensionality reduction"]

# Tag colors
TAG_COLORS = {
    "supervised": "#10b981",
    "unsupervised": "#8b5cf6",
    "regression": "#3b82f6",
    "classification": "#f59e0b",
    "clustering": "#ec4899",
    "dimensionality reduction": "#06b6d4"
}

# Custom CSS
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
    </style>
    """, unsafe_allow_html=True)


def get_tag_style(tag):
    """Get inline style for a tag"""
    color = TAG_COLORS.get(tag, "#6366f1")
    return f"background: {color}22; color: {color}; border: 1px solid {color}44;"


def render_algorithm_card(algo, assets_path):
    """Render a single algorithm card using native Streamlit components"""
    image_path = assets_path / algo["image"]
    
    with st.container():
        # Image
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        
        # Title
        st.markdown(f'<div class="algo-card-title">{algo["name"]}</div>', unsafe_allow_html=True)
        
        # Subtitle if exists
        if "subtitle" in algo:
            st.markdown(f'<div class="algo-card-subtitle">{algo["subtitle"]}</div>', unsafe_allow_html=True)
        
        # Tags
        tags_html = ""
        for tag in algo["tags"]:
            style = get_tag_style(tag)
            tags_html += f'<span class="algo-tag" style="{style}">{tag}</span>'
        st.markdown(f'<div class="algo-card-tags">{tags_html}</div>', unsafe_allow_html=True)
        
        # Strengths
        st.markdown(f'''
        <div class="algo-card-section">
            <div class="algo-card-section-title strengths-title">Strengths</div>
            <div class="algo-card-text">{algo["strengths"]}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Weaknesses
        st.markdown(f'''
        <div class="algo-card-section">
            <div class="algo-card-section-title weaknesses-title">Weaknesses</div>
            <div class="algo-card-text">{algo["weaknesses"]}</div>
        </div>
        ''', unsafe_allow_html=True)


def render_site_header():
    """Render the site title at the very top"""
    st.markdown('''
    <div class="site-header">
        <h1 class="site-title">ML Algorithm Explorer</h1>
        <p class="site-tagline">Find the right machine learning algorithm for your problem</p>
    </div>
    ''', unsafe_allow_html=True)


def render_navigation():
    """Render the navigation menu bar"""
    # Initialize page state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "gallery"
    
    # Add a separator line and navigation wrapper
    st.markdown('<div class="nav-wrapper">', unsafe_allow_html=True)
    
    # Center the navigation using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Try using segmented control (Streamlit 1.33+), fallback to styled buttons
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
            # Fallback to regular buttons in a centered layout
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
    """Render the algorithm gallery page"""
    assets_path = Path(__file__).parent / "assets"
    
    # Page header
    st.markdown('''
    <div class="page-header">
        <div class="page-title">Algorithm Gallery</div>
        <div class="page-subtitle">Explore machine learning algorithms and their trade-offs</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Initialize session state for selected tags (now a list for multi-select)
    if "selected_tags" not in st.session_state:
        st.session_state.selected_tags = []
    
    # Tag options with full, readable names
    tag_options = {
        "supervised": "Supervised",
        "unsupervised": "Unsupervised", 
        "regression": "Regression",
        "classification": "Classification",
        "clustering": "Clustering",
        "dimensionality reduction": "Dimensionality Reduction"
    }
    
    # Multi-select filter with full option names
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
    
    # Filter algorithms - show all if no tags selected, otherwise filter by ALL selected tags (AND logic)
    if not st.session_state.selected_tags:
        filtered_algos = ALGORITHMS
    else:
        filtered_algos = [
            algo for algo in ALGORITHMS
            if all(tag in algo["tags"] for tag in st.session_state.selected_tags)
        ]
    
    # Results count
    st.markdown(f'<div class="results-count">Showing <span>{len(filtered_algos)}</span> of {len(ALGORITHMS)} algorithms</div>', unsafe_allow_html=True)
    
    # Render algorithm cards in a grid
    cols_per_row = 3
    for i in range(0, len(filtered_algos), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(filtered_algos):
                with col:
                    render_algorithm_card(filtered_algos[i + j], assets_path)
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)


def finder_page():
    """Render the ML model finder page (coming soon)"""
    st.markdown("""
    <div class="coming-soon">
        <div class="coming-soon-title">Find My ML Model</div>
        <div class="coming-soon-text">
            An adaptive survey is coming soon. Answer questions about your data, 
            problem type, and constraints to discover the best ML algorithm for your use case.
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    load_css()
    
    # 1. Site header (title) at the very top
    render_site_header()
    
    # 2. Navigation menu bar below title
    render_navigation()
    
    # 3. Page content
    if st.session_state.current_page == "gallery":
        gallery_page()
    else:
        finder_page()


if __name__ == "__main__":
    main()
