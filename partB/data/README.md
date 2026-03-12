# Data README

## Overview

This project uses **no external datasets**. All data is generated programmatically using NumPy inside each notebook. There are no files to download, no manual placement steps, and no internet access required. Running any notebook from top to bottom will generate its own data in the first code cell.

---

## Primary Dataset: Synthetic Imbalanced Gaussian Blobs

### Purpose

This dataset is the main testbed for Tasks 2 and 3.1. It is designed to directly target the paper's core claim: that uniform sampling fails when cluster sizes are imbalanced, and that the coreset's adaptive sampling overcomes this failure.

### Specification

| Parameter | Value |
|---|---|
| Total points (n) | 2,000 |
| Dimensions (d) | 2 |
| Number of clusters (k) | 4 |
| Train / Test split | 80% / 20% (1,600 train, 400 test) |
| Random seed | 42 |

### Cluster Details

| Cluster | Points | Train pts (approx) | Centre (x₁, x₂) | Std dev | Fraction of data |
|---|---|---|---|---|---|
| 1 | 1,200 | 960 | (0.0, 0.0) | 1.0 | 60.0% |
| 2 | 500 | 400 | (6.0, 0.0) | 0.8 | 25.0% |
| 3 | 150 | 120 | (3.0, 5.0) | 0.6 | 7.5% |
| 4 | 150 | 120 | (3.0, −5.0) | 0.6 | 7.5% |

The imbalance ratio between the largest and smallest clusters is **8:1**. This is deliberate — the paper's key failure mode for uniform sampling occurs exactly in this regime, where a random 100-point sample has only a ~9% chance of including representative points from clusters 3 or 4.

### How It Is Generated

```python
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

cluster_sizes = [1200, 500, 150, 150]
centers       = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0], [3.0, -5.0]])
cluster_stds  = [1.0, 0.8, 0.6, 0.6]

parts = [np.random.randn(n, 2) * s + c
         for n, c, s in zip(cluster_sizes, centers, cluster_stds)]
X_all = np.vstack(parts)
np.random.shuffle(X_all)

X_train, X_test = train_test_split(X_all, test_size=0.2, random_state=SEED)
```

Each cluster is generated as an isotropic (spherical) Gaussian blob — this satisfies the paper's ε-semi-spherical assumption (Assumption 1, Section 2) which requires all covariance eigenvalues to lie within [ε, 1/ε].

### Used In

- `task_2_1.ipynb` — Dataset setup, justification, and visualisation
- `task_2_2.ipynb` — Coreset construction and weighted EM reproduction
- `task_2_3.ipynb` — Figure 3 reproduction and result comparison
- `task_3_1.ipynb` — Ablation A (no distance term) and Ablation B (no rough cover)

---

## Secondary Dataset: Anisotropic Gaussian Blobs (Failure Mode)

### Purpose

Used exclusively in Task 3.2 to demonstrate a controlled failure mode. It violates the ε-semi-spherical assumption by using Gaussian components with extreme eigenvalue ratios (up to ~278:1), causing the coreset's Euclidean-based importance weights to be miscalibrated.

### Specification

| Parameter | Value |
|---|---|
| Total points (n) | 1,500 |
| Dimensions (d) | 2 |
| Number of clusters (k) | 3 |
| Train / Test split | 80% / 20% |
| Random seed | 42 |

### Cluster Details

| Cluster | Points | Centre | Covariance | Eigenvalue ratio | Assumption status |
|---|---|---|---|---|---|
| 1 | 800 | (0.0, 0.0) | diag(1.0, 1.0) | 1:1 | ✅ Spherical (satisfied) |
| 2 | 400 | (12.0, 0.0) | diag(100.0, 1.0) | 100:1 | ❌ Violated |
| 3 | 300 | (6.0, 8.0) | [[5,4],[4,5]] rotated | ~278:1 | ❌ Violated |

### How It Is Generated

```python
rng = np.random.default_rng(42)

# Cluster 1: spherical — assumption satisfied
X1 = rng.multivariate_normal([0.0, 0.0], np.eye(2), 800)

# Cluster 2: highly elongated along x-axis (std=10 in x, std=1 in y)
X2 = rng.multivariate_normal([12.0, 0.0], np.diag([100.0, 1.0]), 400)

# Cluster 3: rotated elongated (correlated covariance)
cov3 = np.array([[5.0, 4.0], [4.0, 5.0]])
X3 = rng.multivariate_normal([6.0, 8.0], cov3, 300)
```

### Why It Causes Failure

The coreset construction uses Euclidean distance `dist(x, B)` to compute importance weights. For elongated clusters, Euclidean distance treats all spatial directions equally. A point at the elongated tip of cluster 2 appears geometrically isolated (high Euclidean distance from B) even though it sits well within the cluster's probability mass. This leads to over-sampling of extremes and under-sampling of the narrow axis, producing miscalibrated weights and a degraded coreset.

The coreset's gap to full-data log-likelihood grows from **0.004** (spherical data, assumption satisfied) to **0.113** (anisotropic data, assumption violated) — a 28× increase.

### Used In

- `task_3_2.ipynb` — Failure mode experiment and visualisation

---

## Why Not Use the Paper's Datasets?

The paper's experiments use three real-world datasets:

| Dataset | n | d | k | Availability |
|---|---|---|---|---|
| MNIST | 60,000 | 100 (PCA-reduced) | 10 | Public but requires heavy preprocessing |
| Neural tetrode recordings | 319,209 | 152 | 33 | Not publicly available |
| CSN accelerometer (earthquake) | 40,000 | 17 | 6 | Requires institutional registration |

These were not used for three reasons:

1. **Assignment constraint** — Task 2.1 explicitly requires a dataset that is not the paper's exact dataset but belongs to the same problem type (GMM training).
2. **Computational infeasibility** — Full-data EM on 319,209 × 152-dimensional points would take hours on a CPU-only notebook. The paper used a compiled C++ implementation on an Intel Xeon server.
3. **Reproducibility** — The tetrode and CSN datasets are not freely downloadable in a standard form, making full reproduction impractical for a coursework submission.

The synthetic dataset reproduces the key statistical property that makes the paper interesting (cluster imbalance) while remaining fully transparent, reproducible, and computationally tractable.

---

## Reproducibility Notes

- All data generation uses `SEED = 42` consistently across all notebooks.
- The seed is passed explicitly to both `np.random.seed()` and `np.random.default_rng(seed)` in every notebook to ensure identical results across runs.
- No data files are stored on disk — everything is regenerated from code each time.
- Running `Kernel → Restart & Run All` on any notebook will reproduce the exact same dataset and results.
