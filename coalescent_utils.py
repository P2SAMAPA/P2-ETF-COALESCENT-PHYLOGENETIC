import numpy as np
from scipy.spatial.distance import pdist, squareform

def compute_coalescent_scores(returns, metric="correlation", linkage_method="average"):
    """
    Compute divergence score as the average pairwise distance from each ETF to all others.
    This is a robust proxy for tree-based cophenetic distance.
    """
    returns_clean = returns.dropna(axis=0, how='any')
    returns_clean = returns_clean.dropna(axis=1, how='any')
    n = returns_clean.shape[1]
    if n < 3:
        return {t: 0.0 for t in returns_clean.columns}
    try:
        if metric == "correlation":
            corr = returns_clean.corr().fillna(0)
            dist = 1 - corr
            np.fill_diagonal(dist.values, 0)
            # Average distance from each ETF to all others
            avg_dist = dist.mean(axis=1).values
        elif metric == "euclidean":
            standardized = (returns_clean - returns_clean.mean()) / returns_clean.std()
            dist_matrix = squareform(pdist(standardized.T, metric='euclidean'))
            avg_dist = np.mean(dist_matrix, axis=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        tickers = returns_clean.columns
        return {ticker: avg_dist[i] for i, ticker in enumerate(tickers)}
    except Exception as e:
        print(f"    Warning: coalescent failed ({e}), returning zeros")
        return {t: 0.0 for t in returns_clean.columns}
