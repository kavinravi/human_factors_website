# ML Model Finder – Deterministic Survey Spec

This document defines a **purely rule-based, deterministic** multiple-choice/true–false survey to recommend one or more ML models from your gallery. No LLM is required at runtime; everything can be implemented with if/else logic and simple scoring.

Models covered:

* **Supervised – Regression**

  * Linear Regression
  * Decision Tree Regression
  * Random Forest Regression
  * Gradient Boosting Regression
  * KNN Regression

* **Supervised – Classification**

  * Logistic Regression
  * Decision Tree Classification
  * Random Forest Classification
  * Gradient Boosting Classification
  * SVM
  * KNN Classification
  * Naive Bayes

* **Unsupervised – Clustering**

  * K-Means
  * Gaussian Mixture Models (GMM)
  * DBSCAN
  * Spectral Clustering
  * Hierarchical Agglomerative Clustering

* **Dimensionality Reduction**

  * PCA
  * t-SNE
  * UMAP

---

## 1. High-Level Branching Logic

All users start at **Q1**.

* **Q1** decides the high-level category:

  * Supervised (with target/labels) → **Supervised Branch** (Q2 → …)
  * Unsupervised clustering → **Clustering Branch** (Q3, Q4, Q12 → …)
  * Dimensionality reduction / visualization → **Dimensionality Reduction Branch** (Q3, Q4, Q19 → …)

Some questions are **shared** between branches (dataset size, number of features, etc.), but each branch has its own specific follow-up questions and model mapping rules.

Implementation tip: in Streamlit, store answers in `st.session_state` and only display questions relevant to the chosen branch. The logic below is deterministic: for each answer choice, you either **filter out** or **prioritize** certain models.

---

## 2. Global / Shared Questions

These can be asked once and reused by all branches.

### Q1 – Problem Category

* **Type:** Single-choice multiple choice

* **Question:**

  * Which of the following best describes your problem?

* **Options:**

  1. I have input features and a known outcome (label) I want to predict.
  2. I want to discover natural groups or segments in my data.
  3. I want to reduce the number of features or visualize high-dimensional data.

* **Routing:**

  * (1) → Supervised Branch → ask **Q2**, then **Q3**, **Q4**, **Q5**, **Q7**, **Q8**.
  * (2) → Clustering Branch → ask **Q3**, **Q4**, then **Q12**+.
  * (3) → Dimensionality Reduction Branch → ask **Q3**, **Q4**, then **Q19**+.

---

### Q3 – Dataset Size (Rows)

* **Type:** Single-choice multiple choice

* **Question:**

  * Approximately how many data points (rows) does your dataset have?

* **Options:**

  1. Fewer than 1,000
  2. 1,000 – 10,000
  3. 10,000 – 100,000
  4. 100,000 – 1,000,000
  5. More than 1,000,000

* **Usage (rules, not routing):**

  * Very large (4–5):

    * Penalize or disallow: SVM, KNN (both), t-SNE, Spectral Clustering, Hierarchical Clustering, GMM and DBSCAN on huge n.
    * Prefer: Linear/Logistic Regression, Random Forest, Gradient Boosting (up to mid-large), PCA, UMAP, K-Means.
  * Very small (1):

    * Penalize: Very complex Gradient Boosting, deep trees, very many clusters.
    * Prefer: Simpler models (Linear/Logistic, Naive Bayes, small trees, KNN on tiny n).

---

### Q4 – Number of Features (Columns)

* **Type:** Single-choice multiple choice

* **Question:**

  * Roughly how many features (columns) does your dataset have?

* **Options:**

  1. Fewer than 10
  2. 10 – 100
  3. 100 – 1,000
  4. More than 1,000

* **Usage (rules):**

  * High-dimensional (3–4):

    * Penalize: KNN (both), DBSCAN, K-Means without prior dim reduction.
    * Prefer: Linear/Logistic Regression, tree ensembles, PCA, UMAP before clustering.

---

### Q5 – Real-Time Prediction Requirement

* **Type:** Single-choice (Yes/No)

* **Question:**

  * Do you need very fast predictions in real-time (many predictions per second)?

* **Options:**

  1. Yes, prediction speed is critical.
  2. No, a small delay per prediction is acceptable.

* **Usage (rules, supervised only):**

  * If (1) Yes:

    * Penalize: KNN (both), very large Random Forests, very complex Gradient Boosting, SVM with non-linear kernels.
    * Prefer: Linear/Logistic Regression, Naive Bayes, small trees, compact Gradient Boosting.

---

### Q7 – Priority: Interpretability vs Performance

* **Type:** Single-choice multiple choice

* **Question:**

  * Which is more important for your use case?

* **Options:**

  1. High interpretability (easy to explain to non-experts).
  2. A balance between interpretability and accuracy.
  3. Highest possible predictive performance, even if the model is a black box.

* **Usage (rules):**

  * (1) Interpretability:

    * Regression: Linear Regression, shallow Decision Tree Regression.
    * Classification: Logistic Regression, Decision Tree Classification, Naive Bayes.
    * Clustering: K-Means, Hierarchical Clustering (dendrogram).
    * Dim-reduction: PCA.
  * (2) Balance:

    * Regression: Random Forest Regression, KNN Regression.
    * Classification: Random Forest, simpler Gradient Boosting, KNN.
    * Clustering: GMM, K-Means, DBSCAN (explanations at a higher level).
    * Dim-reduction: UMAP.
  * (3) Performance:

    * Regression: Gradient Boosting Regression, Random Forest Regression.
    * Classification: Gradient Boosting Classification, Random Forest, SVM.
    * Clustering: Spectral Clustering (small n), GMM, DBSCAN (when appropriate).
    * Dim-reduction: t-SNE / UMAP for visualization.

---

### Q8 – Robustness to Outliers / Noise

* **Type:** Single-choice (Yes/No)

* **Question:**

  * Do you expect many outliers or noisy points in your data?

* **Options:**

  1. Yes, there are many outliers/noisy points.
  2. No, the data is mostly clean with few outliers.

* **Usage (rules):**

  * If (1) Yes:

    * Prefer: Tree-based models, Random Forest, Gradient Boosting, DBSCAN, robust clustering methods.
    * Penalize: K-Means, Linear/Logistic Regression without robustification, SVM with sensitive kernels.

---

## 3. Supervised Branch

Users who choose **Option 1** in Q1 follow this branch.

### Q2 – Target Variable Type

* **Type:** Single-choice multiple choice

* **Question:**

  * What best describes your target (output) variable?

* **Options:**

  1. A numeric value with no strict 0–1 bounds (e.g., price, temperature, time).
  2. A numeric proportion or rate strictly between 0 and 1 (e.g., conversion rate, click-through rate).
  3. A binary category (two classes, e.g., spam vs not spam).
  4. A categorical label with more than two classes (e.g., dog/cat/bird).

* **Routing:**

  * (1) → **Regression Sub-Branch**, proceed to **Q3**, **Q4**, **Q5**, **Q7**, **Q8**.
  * (2) → **Regression Sub-Branch (Proportion)**, same as above but Beta Regression considered.
  * (3) → **Classification Sub-Branch (Binary)**, proceed to **Q3**, **Q4**, **Q5**, **Q6**, **Q7**, **Q8**, **Q9**.
  * (4) → **Classification Sub-Branch (Multiclass)**, proceed to **Q3**, **Q4**, **Q5**, **Q6**, **Q7**, **Q8**, **Q9**.

---

### Q6 – Data Type (Text vs General)

* **Type:** Single-choice (Yes/No)

* **Question:**

  * Is your input data primarily text (e.g., emails, reviews, documents) represented as word counts, TF-IDF, or similar?

* **Options:**

  1. Yes, the inputs are primarily text.
  2. No, the inputs are mainly numeric/categorical features.

* **Usage (classification only):**

  * If (1) Yes → Strongly favor **Naive Bayes**, Logistic Regression; penalize KNN and tree ensembles slightly.

---

### Q9 – Class Balance

* **Type:** Single-choice (Yes/No)

* **Question:**

  * Are your classes highly imbalanced (for example, one class is much rarer than the others)?

* **Options:**

  1. Yes, the classes are highly imbalanced.
  2. No, the classes are reasonably balanced.

* **Usage (classification only):**

  * If (1) Yes:

    * Prefer: Gradient Boosting, Random Forest (with class weights), Logistic Regression (with class weights), SVM (with class weights).
    * Penalize: Plain KNN, plain Naive Bayes without additional handling.

---

### 3.1 Regression Sub-Branch – Model Rules

**Applies when Q2 = 1 or 2.** Questions used: Q2, Q3, Q4, Q5, Q7, Q8.

For each model, define **hard filters** and **preferences**.

#### Linear Regression (Regression)

* **Hard filters:**

  * Q2 must be (1) numeric unbounded (not pure proportion model).
* **Preferred when:**

  * Q7 = 1 (interpretability).
  * Q3 ≥ 2 (enough data).
  * Q8 = 2 (few outliers).
  * Q5 = 1 or 2 (always fast at prediction).

#### Decision Tree Regression

* **Hard filters:**

  * Q3 ≥ 1 (works at any size, but may overfit on tiny n).
* **Preferred when:**

  * Q7 = 1 or 2 (interpretability / balance).
  * Q8 = 1 (needs robustness to outliers).
  * No extreme need for the very best performance.

#### Random Forest Regression

* **Hard filters:**

  * Q3 ≥ 2 (≥ 1,000 rows is ideal; works below but less stable).
* **Preferred when:**

  * Q7 = 2 or 3 (balance / performance).
  * Q8 = 1 (outliers & noise).
  * Q5 = 2 (real-time not super strict, but still reasonable speed).

#### Gradient Boosting Regression

* **Hard filters:**

  * Q3 ≥ 2 (at least 1,000 rows).
* **Preferred when:**

  * Q7 = 3 (maximum performance).
  * Q5 = 2 (does not require ultra-fast prediction in all settings).
  * Q8 = 2 (cleaner data is ideal, though boosting can be robust with tuning).

#### KNN Regression

* **Hard filters:**

  * Q3 ∈ {1, 2} (fewer than 10,000 rows ideally).
  * Q4 ∈ {1, 2} (fewer than 100 features ideally).
* **Preferred when:**

  * Q7 = 2 (balance between interpretability and performance at small scale).
  * Q5 = 2 (no strict real-time requirement).

**Regression Recommendation Procedure (deterministic):**

1. Filter models based on Q2, Q3, Q4 hard rules.
2. Among remaining, sort by:

   * Compatibility with Q7 (interpretability vs performance).
   * Compatibility with Q8 (outliers).
   * Compatibility with Q5 (prediction speed).
3. Present top 2–3 regression models to the user.

---

### 3.2 Classification Sub-Branch – Model Rules

**Applies when Q2 = 3 (binary) or Q2 = 4 (multiclass).** Questions used: Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9.

#### Logistic Regression

* **Hard filters:**

  * Q2 ∈ {3, 4} (binary or multiclass with one-vs-rest/multinomial).
* **Preferred when:**

  * Q6 = 1 (text), or Q7 = 1 (interpretability).
  * Real-time needed: Q5 = 1 (very fast predictions).
  * Class imbalance: Q9 = 1 (with class weights).

#### Decision Tree Classification

* **Hard filters:**

  * None (works in most settings).
* **Preferred when:**

  * Q7 = 1 or 2 (interpretability / balance).
  * Q8 = 1 (outliers).

#### Random Forest Classification

* **Hard filters:**

  * Q3 ≥ 2 (at least 1,000 rows ideal).
* **Preferred when:**

  * Q7 = 2 or 3 (balance / performance).
  * Q8 = 1 (outliers & noisy features).
  * Q9 = 1 (imbalanced, with class weights).

#### Gradient Boosting Classification

* **Hard filters:**

  * Q3 ≥ 2.
* **Preferred when:**

  * Q7 = 3 (performance).
  * Q9 = 1 (imbalance, using built-in options).
  * Q5 = 2 (prediction speed less critical).

#### SVM (Classification)

* **Hard filters:**

  * Q3 ≤ 3 (≤ 100k rows); prefer smaller (< 10k).
* **Preferred when:**

  * Q7 = 3 (performance) and Q4 ∈ {1, 2} (not extremely high dimensional unless linear).
  * Q5 = 2 (no hard real-time requirement).

#### KNN Classification

* **Hard filters:**

  * Q3 ∈ {1, 2} (up to 10k rows).
  * Q4 ∈ {1, 2} (up to 100 features).
* **Preferred when:**

  * Q7 = 2 (balance).
  * Q5 = 2 (prediction speed not critical).

#### Naive Bayes

* **Hard filters:**

  * None, but especially suitable when Q6 = 1 (text data).
* **Preferred when:**

  * Q6 = 1 (text input).
  * Q7 = 1 or 2 (simple, interpretable-ish, good baseline).
  * Q5 = 1 (needs very fast training and prediction).

**Classification Recommendation Procedure (deterministic):**

1. Filter by Q2 (binary/multiclass – all listed support both via strategies; no strong filter needed here).
2. Apply hard filters for Q3, Q4, Q5, Q6.
3. Rank remaining models by:

   * Alignment with Q7 (interpretability vs performance).
   * Alignment with Q8 (outliers) – trees/ensembles if many outliers.
   * Alignment with Q9 (class imbalance).
4. Present top 2–3 classification models.

---

## 4. Clustering Branch

Users who choose **Option 2** in Q1 follow this branch. Questions used: Q3, Q4, Q7, Q8, Q12–Q18.

### Q12 – Knowledge of Number of Clusters

* **Type:** Single-choice (Yes/No)
* **Question:**

  * Do you have a reasonable guess for the number of clusters/groups you expect?
* **Options:**

  1. Yes, I have a rough idea of how many clusters there are.
  2. No, I do not know how many clusters there are.

### Q13 – Outlier Detection Importance

* **Type:** Single-choice (Yes/No)
* **Question:**

  * Is explicitly identifying outliers/noise points important for your task?
* **Options:**

  1. Yes, I want to detect and separate outliers/noise.
  2. No, I mainly care about the main clusters.

### Q14 – Cluster Shape & Density

* **Type:** Single-choice multiple choice
* **Question:**

  * Which description best matches the clusters you expect?
* **Options:**

  1. Roughly spherical or compact “blob-like” clusters of similar size.
  2. Arbitrarily shaped clusters (e.g., chains, rings) or clusters of very different densities.
  3. I am not sure.

### Q15 – Soft vs Hard Cluster Assignments

* **Type:** Single-choice (Yes/No)
* **Question:**

  * Do you want a probability for each point belonging to each cluster (soft assignments)?
* **Options:**

  1. Yes, I want probabilistic cluster memberships.
  2. No, a single cluster label per point is enough.

### Q16 – Hierarchical Structure

* **Type:** Single-choice (Yes/No)
* **Question:**

  * Is it important to see a hierarchy of clusters (e.g., via a tree/dendrogram) rather than a single flat clustering?
* **Options:**

  1. Yes, I want to see clusters at multiple levels of granularity.
  2. No, a single flat partition is enough.

### Q17 – Dataset Size Constraint for Heavy Algorithms

* **Type:** Single-choice (Yes/No)
* **Question:**

  * Is your dataset small enough that slower, more computationally intensive algorithms are acceptable?
* **Options:**

  1. Yes, my dataset is small (for example, fewer than ~10,000 points).
  2. No, my dataset is larger, and I need methods that scale well.

> Note: This essentially refines Q3 for clustering, to decide between things like Spectral Clustering vs K-Means.

### Q18 – Allowing Unclassified Points (Noise)

* **Type:** Single-choice (Yes/No)
* **Question:**

  * Are you okay with some data points being left unassigned to any cluster (marked as noise)?
* **Options:**

  1. Yes, I am okay with some points being left unclassified as noise.
  2. No, I want every point assigned to some cluster.

---

### 4.1 Clustering Models – Rules

#### K-Means

* **Hard filters:**

  * Ideally Q3 ≤ 5 (scales well, but extremely huge n may still be an issue in practice).
  * Q14 ∈ {1, 3} (spherical or not sure).
* **Preferred when:**

  * Q12 = 1 (known number of clusters).
  * Q7 = 1 or 2 (interpretability / balance).
  * Q13 = 2 (outliers not critical).

#### Gaussian Mixture Models (GMM)

* **Hard filters:**

  * Q3 not extremely large (e.g., ≤ 4 as a soft limit).
* **Preferred when:**

  * Q12 = 1 (number of components known or guessed).
  * Q14 ∈ {1, 3} (elliptical clusters).
  * Q15 = 1 (soft assignments needed).

#### DBSCAN

* **Hard filters:**

  * Q3 not extremely large; works best with moderate n and lower dimensions (Q4 ∈ {1, 2}).
  * Q18 != 2 (if the user is not okay with noise points, exclude DBSCAN).
* **Preferred when:**

  * Q13 = 1 (explicit outlier/noise detection is important).
  * Q14 = 2 (arbitrary shapes / varied densities).
  * Q12 = 2 (do not know number of clusters).

#### Spectral Clustering

* **Hard filters:**

  * Q17 = 1 (dataset small; heavy O(n^3)).
* **Preferred when:**

  * Q14 = 2 (complex, non-convex clusters).
  * Q7 = 3 (focus on cluster quality over scalability/interpretability).

#### Hierarchical Agglomerative Clustering

* **Hard filters:**

  * Q3 not extremely large (ideally ≤ 3).
* **Preferred when:**

  * Q16 = 1 (need hierarchy/dendrogram).
  * Q7 = 1 or 2 (interpretability via dendrogram).
  * Q12 can be 2 (you can choose K after seeing the dendrogram).

**Clustering Recommendation Procedure:**

1. Use Q3, Q4, Q17 to remove models that cannot scale.
2. Use Q12–Q16 to filter:

   * Want hierarchy → include Hierarchical.
   * Need soft probabilities → include GMM.
   * Want outlier detection and complex shapes → include DBSCAN.
   * Known K, simple blobs → include K-Means (and possibly GMM).
   * Small n & complex shapes → consider Spectral.
3. Use Q7 and Q8 to adjust ranking (interpretability and outlier handling).
4. Present top 2–3 clustering models.

---

## 5. Dimensionality Reduction Branch

Users who choose **Option 3** in Q1 follow this branch. Questions used: Q3, Q4, Q7, Q19–Q23.

### Q19 – Primary Purpose of Dimensionality Reduction

* **Type:** Single-choice multiple choice
* **Question:**

  * What is the main reason you want to reduce dimensionality?
* **Options:**

  1. General preprocessing / noise reduction before other models.
  2. 2D/3D visualization of the data to see clusters/patterns.
  3. Both preprocessing and visualization are important.

### Q20 – Need to Transform New Data Points

* **Type:** Single-choice (Yes/No)
* **Question:**

  * Will you need to apply the same dimensionality reduction mapping to new data points later?
* **Options:**

  1. Yes, I will need to transform new points consistently.
  2. No, reducing a fixed dataset once is enough.

### Q21 – Focus on Global vs Local Structure

* **Type:** Single-choice multiple choice
* **Question:**

  * What do you care about more in the reduced space?
* **Options:**

  1. Preserving overall global variance/structure.
  2. Preserving small-scale local neighborhoods (local clusters).
  3. A balance between global structure and local neighborhoods.

### Q22 – Dataset Size vs Heavy Visualization Methods

* **Type:** Single-choice (Yes/No)
* **Question:**

  * Is your dataset small or medium-sized so that slower visualization methods are acceptable?
* **Options:**

  1. Yes, my dataset is small/medium (e.g., fewer than ~50,000 points).
  2. No, my dataset is large and I need faster methods.

### Q23 – Determinism Requirement

* **Type:** Single-choice (Yes/No)
* **Question:**

  * Is it important that you get the exact same result every time you run dimensionality reduction with the same data?
* **Options:**

  1. Yes, I need deterministic results.
  2. No, slight randomness between runs is acceptable.

---

### 5.1 Dimensionality Reduction Models – Rules

#### PCA

* **Hard filters:**

  * None; works in most numeric settings.
* **Preferred when:**

  * Q19 ∈ {1, 3} (preprocessing and/or both).
  * Q21 = 1 (global structure).
  * Q23 = 1 (determinism required).
  * Q3 large (PCA scales relatively well).

#### t-SNE

* **Hard filters:**

  * Q22 = 1 (small/medium dataset).
* **Preferred when:**

  * Q19 ∈ {2, 3} (visualization is key).
  * Q21 = 2 (local neighborhood structure more important).
  * Q20 = 2 (no need to transform new points easily).
  * Q23 = 2 (non-determinism acceptable).

#### UMAP

* **Hard filters:**

  * None strict, but works best with small–large datasets; can be heavy on huge data.
* **Preferred when:**

  * Q19 ∈ {2, 3} (visualization and/or both), especially when Q20 = 1 (can transform new points).
  * Q21 = 3 (balance between global and local).
  * Q22 = 1 or 2 (more scalable than t-SNE).
  * Q23 = 2 (non-strict determinism).

**Dimensionality Reduction Recommendation Procedure:**

1. Use Q22 and Q3 to decide if t-SNE is feasible (small/medium n).
2. Use Q19, Q21, Q23 to select between PCA, t-SNE, UMAP:

   * Preprocessing + determinism + global variance → PCA.
   * Visualization + local clusters + small n → t-SNE.
   * Visualization + ability to transform new points + balance global/local → UMAP.
3. Present 1–2 recommended dim-reduction methods.

---

## 6. Putting It All Together

For your Streamlit implementation:

1. Ask **Q1** to pick the branch.
2. Ask global questions (**Q3**, **Q4**) once.
3. Ask branch-specific questions:

   * Supervised: Q2, Q5, Q6 (classification only), Q7, Q8, Q9.
   * Clustering: Q12–Q17, plus reuse Q3, Q4, Q7, Q8.
   * Dim-reduction: Q19–Q23, plus reuse Q3, Q4, Q7.
4. Encode the **hard filters** and **preference rules** as deterministic if/else logic.
5. Output a **ranked list of 1–3 candidate models** with a short justification, using the model strengths/weaknesses you already drafted.

This spec gives you a fully deterministic survey flow with only multiple-choice / true–false questions and clear, rule-based narrowing from high-level problem type down to specific model families.
