import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cophenet

def distance_matrix(returns, metric="correlation"):
    """Return condensed distance matrix."""
    if metric == "correlation":
        corr = returns.corr().fillna(0)
        dist = 1 - corr
        np.fill_diagonal(dist.values, 0)
        condensed = squareform(dist.values, checks=False)
    elif metric == "euclidean":
        standardized = (returns - returns.mean()) / returns.std()
        condensed = pdist(standardized.T, metric='euclidean')
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return condensed

def compute_coalescent_scores(returns, metric="correlation", linkage_method="average"):
    # Clean data: drop rows with any NaN, then drop columns with any NaN
    returns_clean = returns.dropna(axis=0, how='any')
    returns_clean = returns_clean.dropna(axis=1, how='any')
    n = returns_clean.shape[1]
    if n < 3:
        # Not enough assets to build a meaningful tree
        return {t: 0.0 for t in returns_clean.columns}
    try:
        condensed_dist = distance_matrix(returns_clean, metric=metric)
        Z = linkage(condensed_dist, method=linkage_method, metric='precomputed')
        cophenet_dist, _ = cophenet(Z, condensed_dist)
        cophenet_square = squareform(cophenet_dist)
        scores = np.mean(cophenet_square, axis=1)
        tickers = returns_clean.columns
        return {ticker: scores[i] for i, ticker in enumerate(tickers)}
    except Exception as e:
        print(f"    Warning: coalescent failed ({e}), returning zeros")
        return {t: 0.0 for t in returns_clean.columns}
