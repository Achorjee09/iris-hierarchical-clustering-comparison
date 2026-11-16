# Iris Hierarchical Clustering: HAC vs. DIANA

A Python comparison of **agglomerative** and **divisive** hierarchical clustering techniques on the Iris dataset. The project visualizes cluster hierarchies, evaluates clustering quality, and compares execution times.

## Algorithms compared

| Method | Approach | Description |
| --- | --- | --- |
| Single HAC | Agglomerative | Merges clusters using the smallest pairwise distance. |
| Complete HAC | Agglomerative | Merges clusters using the largest pairwise distance. |
| Average HAC | Agglomerative | Merges clusters using average pairwise distance. |
| DIANA HDC | Divisive | Starts with all samples together and recursively splits clusters using a custom DIANA-style procedure. |

## Features

- Standardizes the numeric Iris features using `StandardScaler`.
- Builds and plots a dendrogram for each method.
- Extracts flat clusters (target: `K = 3`).
- Reports **cophenetic correlation**, **silhouette score**, **adjusted Rand index (ARI)** when species labels are present, the number of clusters, and runtime.
- Compares results with bar plots and **2D PCA scatter plots**.
- Benchmarks **Average HAC** versus the custom **DIANA HDC** implementation across sample sizes of 30, 60, 90, 120, and 150.
- Prints cluster-versus-species contingency tables when the dataset includes a `Species` column.

## Project structure

```text
iris-hierarchical-clustering-comparison/
├── hierarchical_clustering_iris.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup and run

Requires Python 3.10 or later.

```bash
git clone https://github.com/YOUR_USERNAME/iris-hierarchical-clustering-comparison.git
cd iris-hierarchical-clustering-comparison
python -m pip install -r requirements.txt
python hierarchical_clustering_iris.py
```

On the first run, the script attempts to download `Iris.csv` from a Google Drive file using `gdown`. If access to that file fails, place an appropriate Iris CSV file named `Iris.csv` in the project directory. The expected dataset has numeric flower-measurement columns and preferably a `Species` column; an `Id` column, if present, is excluded from clustering.

## Evaluation metrics

| Metric | Interpretation |
| --- | --- |
| Cophenetic correlation | How well dendrogram cophenetic distances preserve the original pairwise distances. Higher is better. |
| Silhouette score | Measures cohesion and separation of predicted clusters. Higher is better. |
| Adjusted Rand index (ARI) | Agreement of the cluster assignments with the ground-truth species labels, adjusted for chance. Higher is better; labels are **not** used to fit the clusters. |
| Runtime (ms) | Median elapsed execution time from five runs, measured with `time.perf_counter()`. Lower is faster for the measured execution. |

## Visualizations

The script displays four sets of plots:

1. Four dendrograms (single, complete, average, and DIANA).
2. Bar charts comparing cophenetic correlation, silhouette score, and runtime.
3. PCA scatter plots showing each method's cluster labels.
4. A runtime comparison plot across subsample sizes.

## Notes and limitations

- This project uses a **custom educational DIANA-style implementation**, not a library-validated canonical implementation. Its results should be interpreted accordingly.
- `fcluster(..., criterion="maxclust")` may return **fewer than `K` clusters** when dendrogram heights tie. Check the printed `Clusters` column.
- The dashed dendrogram threshold is a visual guide and may not exactly match the actual flat clustering in tie cases.
- The timing comparison reflects these specific Python implementations, not an algorithm-independent complexity comparison: SciPy's HAC code is optimized, while the DIANA routine uses Python recursion.
- Runtime varies by hardware, software versions, and current system load.
- PCA is used for 2D visualization, **not** for fitting the clusters.
- The dataset is downloaded from an external Google Drive source; verify its provenance and usage terms before redistributing it.

## Technologies

Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib, and gdown.

## License

No license is included by default. If you own the code and want others to reuse it, add a license such as MIT using GitHub's license chooser. Check the dataset's rights separately.
