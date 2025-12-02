# Supervised

## Regression

1. **Linear Regression**
   - *Strengths*: Simple, highly interpretable, very fast to train and predict; coefficients directly show feature importance and direction
   - *Weaknesses*: Assumes linear relationships; sensitive to outliers; struggles with multicollinearity; cannot capture complex patterns without manual feature engineering

2. **Decision Trees**
   - *Strengths*: Highly interpretable (visualizable); handles non-linear relationships naturally; no feature scaling required; handles mixed data types
   - *Weaknesses*: Prone to overfitting; unstable (small data changes cause large tree changes); poor extrapolation beyond training data range

3. **Random Forest**
   - *Strengths*: Robust to overfitting compared to single trees; handles non-linearity and interactions well; provides feature importance; works well out-of-the-box with minimal tuning
   - *Weaknesses*: Less interpretable than single trees; slower training/prediction than linear models; memory-intensive for large forests; still struggles with extrapolation

4. **Gradient Boosting Models** (XGBoost, LightGBM, CatBoost)
   - *Strengths*: Often achieves state-of-the-art accuracy on tabular data; handles non-linearity and interactions; can handle missing values (implementation-dependent); feature importance available
   - *Weaknesses*: Requires careful hyperparameter tuning; prone to overfitting without regularization; slower to train than random forests; less interpretable


5. **K-Nearest Neighbors (KNN) Regression**
   - *Strengths*: Simple and intuitive; no training phase (lazy learner); naturally captures non-linear relationships; no assumptions about data distribution
   - *Weaknesses*: Slow at prediction time for large datasets (must search all points); sensitive to feature scaling; suffers in high dimensions (curse of dimensionality); requires meaningful distance metric; no model to interpret

## Classification

1. **Logistic Regression**
   - *Strengths*: Simple, fast, and highly interpretable; outputs well-calibrated probabilities; works well with linearly separable data; easily extends to multiclass (one-vs-rest, multinomial)
   - *Weaknesses*: Assumes linear decision boundary; struggles with complex non-linear patterns; sensitive to outliers and multicollinearity

2. **Decision Trees**
   - *Strengths*: Highly interpretable; handles non-linear boundaries naturally; no feature scaling needed; outputs class probabilities; handles multiclass natively
   - *Weaknesses*: Prone to overfitting; high variance (unstable); can create biased splits with imbalanced classes

3. **Random Forest**
   - *Strengths*: Robust to overfitting; handles high-dimensional data well; provides feature importance and OOB error estimates; works well with imbalanced data (with class weighting)
   - *Weaknesses*: Less interpretable than single trees; can be slow for real-time prediction with many trees; biased toward features with many categories

4. **Gradient Boosting Models** (XGBoost, LightGBM, CatBoost)
   - *Strengths*: Often best-in-class accuracy on tabular data; handles class imbalance well; built-in regularization options; can handle missing values
   - *Weaknesses*: Requires careful tuning to avoid overfitting; computationally expensive to train; less interpretable; sensitive to noisy labels

5. **Support Vector Machines (SVM)**
   - *Strengths*: Effective in high-dimensional spaces; works well with clear margin of separation; kernel trick enables non-linear boundaries; memory-efficient (uses support vectors only)
   - *Weaknesses*: Computationally expensive for large datasets (O(n²) to O(n³)); sensitive to feature scaling; requires kernel selection and tuning; probability estimates require additional computation (Platt scaling); binary by nature (multiclass requires one-vs-one or one-vs-rest)

6. **K-Nearest Neighbors (KNN) Classification**
   - *Strengths*: Simple and intuitive; no training phase (lazy learner); naturally handles multiclass; decision boundaries can be arbitrarily complex; no assumptions about data distribution
   - *Weaknesses*: Slow at prediction time for large datasets (must search all points); sensitive to feature scaling and irrelevant features; suffers in high dimensions (curse of dimensionality); requires meaningful distance metric; no model to interpret

7. **Naive Bayes**
   - *Strengths*: Extremely fast to train and predict; works well with high-dimensional data (e.g., text); handles multiclass natively; performs surprisingly well despite strong independence assumption; good baseline for text classification
   - *Weaknesses*: Assumes feature independence (rarely true in practice); poor probability estimates (often overconfident); sensitive to feature representation; struggles when features are correlated

---

# Unsupervised (Clustering)

1. **K-Means**
   - *Strengths*: Simple and fast (O(n)); scales well to large datasets; works well with spherical, evenly-sized clusters; easy to interpret
   - *Weaknesses*: Must specify K in advance; assumes spherical clusters of similar size; sensitive to initialization and outliers; only finds convex clusters

2. **Gaussian Mixture Models (GMM)**
   - *Strengths*: Soft clustering (probabilistic assignments); can model elliptical clusters of different shapes/sizes; provides density estimation; handles overlapping clusters
   - *Weaknesses*: Must specify number of components; sensitive to initialization; assumes Gaussian distributions; computationally more expensive than K-Means; can converge to local optima

3. **DBSCAN**
   - *Strengths*: No need to specify number of clusters; finds arbitrarily shaped clusters; robust to outliers (marks them as noise); only two hyperparameters (eps, min_samples)
   - *Weaknesses*: Struggles with varying density clusters; sensitive to hyperparameter choices; not suitable for high-dimensional data without preprocessing; no soft assignments

4. **Spectral Clustering**
   - *Strengths*: Finds non-convex, complex-shaped clusters; based on graph connectivity rather than distance; works well when cluster structure is defined by connectivity
   - *Weaknesses*: Computationally expensive (O(n³) for eigendecomposition) — not suitable for large datasets; must specify number of clusters; sensitive to similarity graph construction

5. **Hierarchical Agglomerative Clustering**
   - *Strengths*: Produces interpretable dendrogram; no need to pre-specify K (can cut at any level); can use various linkage methods (single, complete, average, Ward)
   - *Weaknesses*: Computationally expensive (O(n²) memory, O(n³) time for naive implementation); no reassignment after merging (greedy); sensitive to noise and outliers; doesn't scale well

---

# Dimensionality Reduction

1. **Principal Component Analysis (PCA)**
   - *Strengths*: Fast and deterministic; preserves global variance structure; components are orthogonal and ranked by importance; useful for noise reduction and preprocessing
   - *Weaknesses*: Only captures linear relationships; sensitive to feature scaling; components can be hard to interpret; may lose important local structure

2. **T-SNE** (t-Distributed Stochastic Neighbor Embedding)
   - *Strengths*: Excellent for 2D/3D visualization; preserves local neighborhood structure; reveals clusters and patterns invisible to PCA
   - *Weaknesses*: Computationally expensive (O(n²)); non-deterministic (results vary across runs); perplexity hyperparameter sensitive; distances between clusters are not meaningful; not suitable for dimensionality reduction beyond visualization; cannot transform new data points

3. **UMAP** (Uniform Manifold Approximation and Projection)
   - *Strengths*: Faster than T-SNE; better preserves global structure while maintaining local structure; scales to larger datasets; can be used for general dimensionality reduction (not just visualization); supports transforming new data points
   - *Weaknesses*: Hyperparameter sensitive (n_neighbors, min_dist); non-deterministic; less theoretical foundation than PCA; can distort densities; results require careful interpretation
