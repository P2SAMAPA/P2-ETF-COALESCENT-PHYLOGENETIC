import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cophenet

def distance_matrix(returns, metric="correlation"):
    """
    Compute pairwise distance between ETFs.
    metric: 'correlation' (1 - Pearson correlation) or 'euclidean' on standardized returns.
    Returns condensed distance matrix (pdist format).
    """
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
    """
    Build UPGMA tree from distance matrix, compute cophenetic distances,
    and return for each ETF the average distance to all others (divergence score).
    """
    returns_clean = returns.dropna()
    n = returns_clean.shape[1]
    if returns_clean.shape[0] < n:
        # Not enough data: return zeros
        return {t: 0.0 for t in returns_clean.columns}
    condensed_dist = distance_matrix(returns_clean, metric=metric)
    Z = linkage(condensed_dist, method=linkage_method, metric='precomputed')
    # Cophenetic distance matrix (full squareform)
    cophenet_dist, _ = cophenet(Z, condensed_dist)
    # Convert to square form
    cophenet_square = squareform(cophenet_dist)
    # For each leaf, average distance to all other leaves
    scores = np.mean(cophenet_square, axis=1)
    tickers = returns_clean.columns
    return {ticker: scores[i] for i, ticker in enumerate(tickers)}
