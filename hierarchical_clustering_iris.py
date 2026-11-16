import os, time, gdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

if not os.path.exists("Iris.csv"):
    gdown.download(id="1jGb9KIM-9kNAILv7ObH2rKpCMjNY_leK", output="Iris.csv")

df = pd.read_csv("Iris.csv")
X = StandardScaler().fit_transform(
    df.select_dtypes("number").loc[:, lambda d: d.columns.str.lower() != "id"]
)
y = df["Species"].to_numpy() if "Species" in df else None
dist = pdist(X)
K = 3

def diana(X):
    D = squareform(pdist(X))
    n, nodes = len(X), []
    def split(idx):
        if len(idx) == 1:
            return ("leaf", idx[0])
        R = list(idx)
        S = [R.pop(np.argmax(D[np.ix_(R, R)].mean(axis=1)))]
        while len(R) > 1:
            diff = (D[np.ix_(R, R)].sum(axis=1) / (len(R)-1)
                    - D[np.ix_(R, S)].mean(axis=1))
            if max(diff) <= 0:
                break
            S.append(R.pop(np.argmax(diff)))
        left, right = split(S), split(R)
        nodes.append((D[np.ix_(idx, idx)].max(), len(idx), left, right))
        return ("node", len(nodes)-1)
    split(list(range(n)))
    order = sorted(range(n-1), key=lambda i: nodes[i][:2])
    ids = {i: n+j for j, i in enumerate(order)}
    def cid(node):
        return node[1] if node[0] == "leaf" else ids[node[1]]
    return np.array([[cid(nodes[i][2]), cid(nodes[i][3]),
                      nodes[i][0], nodes[i][1]] for i in order], dtype=float)

methods = {"Single HAC": "single", "Complete HAC": "complete",
           "Average HAC": "average", "DIANA HDC": None}
Z, times = {}, {}
for name, method in methods.items():
    runs = []
    for _ in range(5):
        start = time.perf_counter()
        z = linkage(X, method=method) if method else diana(X)
        runs.append((time.perf_counter()-start)*1000)
    Z[name], times[name] = z, np.median(runs)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, (name, z) in zip(axes.flat, Z.items()):
    h = (z[-K, 2] + z[-K+1, 2]) / 2
    dendrogram(z, ax=ax, no_labels=True, color_threshold=h)
    ax.axhline(h, color="red", ls="--", label=f"k={K}")
    ax.set(title=name, xlabel="Samples", ylabel="Distance")
    ax.legend()
plt.tight_layout()
plt.show()

rows, labels = [], {}
for name, z in Z.items():
    lab = fcluster(z, K, criterion="maxclust")
    labels[name] = lab
    rows.append([name, cophenet(z, dist)[0],
                 silhouette_score(X, lab) if 1 < len(set(lab)) < len(X) else np.nan,
                 adjusted_rand_score(y, lab) if y is not None else np.nan,
                 len(set(lab)), times[name]])
results = pd.DataFrame(rows, columns=[
    "Method", "Cophenetic", "Silhouette", "ARI", "Clusters", "Time(ms)"
]).set_index("Method")
print("\nEvaluation Results:")
print(results.round(4))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["Cophenetic", "Silhouette", "Time(ms)"]):
    results[col].plot.bar(ax=ax, title=col, rot=20)
plt.tight_layout()
plt.show()

pca = PCA(n_components=2)
P = pca.fit_transform(X)
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
for ax, (name, lab) in zip(axes, labels.items()):
    ax.scatter(P[:, 0], P[:, 1], c=lab, cmap="viridis", s=20)
    ax.set_title(name)
    ax.set_xlabel("PC1")
axes[0].set_ylabel("PC2")
plt.tight_layout()
plt.show()

sizes = [30, 60, 90, 120, 150]
hac_t, hdc_t = [], []
rng = np.random.default_rng(42)
for s in sizes:
    sub = X[rng.choice(len(X), s, replace=False)]
    for method, record in [("average", hac_t), (None, hdc_t)]:
        runs = []
        for _ in range(5):
            start = time.perf_counter()
            linkage(sub, method=method) if method else diana(sub)
            runs.append((time.perf_counter()-start)*1000)
        record.append(np.median(runs))
plt.plot(sizes, hac_t, "o-", label="HAC (Average)")
plt.plot(sizes, hdc_t, "s-", label="HDC (DIANA)")
plt.xlabel("Number of samples")
plt.ylabel("Runtime (ms)")
plt.title("Runtime Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

if y is not None:
    for name, lab in labels.items():
        print(f"\n{name}:")
        print(pd.crosstab(pd.Series(lab, name="Cluster"),
                          pd.Series(y, name="Species")))

print("\n===== SUMMARY =====")
print("Best Cophenetic:", results["Cophenetic"].idxmax())
print("Best Silhouette:", results["Silhouette"].idxmax())
print("Fastest:", results["Time(ms)"].idxmin())
if y is not None:
    print("Best ARI:", results["ARI"].idxmax())
