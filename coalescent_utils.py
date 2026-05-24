import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, to_tree

def distance_matrix(returns, metric="correlation"):
    """
    Compute pairwise distance between ETFs.
    metric: 'correlation' (1 - Pearson correlation) or 'euclidean' on standardized returns.
    Returns condensed distance matrix (pdist format).
    """
    # returns: DataFrame (T x n)
    if metric == "correlation":
        # correlation distance = 1 - corrcoef
        corr = returns.corr().fillna(0)
        dist = 1 - corr
        # Ensure symmetry and zero diagonal
        np.fill_diagonal(dist.values, 0)
        # Convert to condensed form
        condensed = squareform(dist.values, checks=False)
    elif metric == "euclidean":
        # Standardize each ETF's returns (z-score) then Euclidean
        standardized = (returns - returns.mean()) / returns.std()
        condensed = pdist(standardized.T, metric='euclidean')
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return condensed

def leaf_depths_from_linkage(linkage_matrix, n):
    """
    Convert linkage matrix to a tree and compute root-to-leaf depth (sum of branch lengths).
    Returns a list of depth for each original leaf.
    """
    tree = to_tree(linkage_matrix, rd=False)
    depths = [0.0] * n
    def _traverse(node, current_depth):
        if node.is_leaf():
            depths[node.get_id()] = current_depth
        else:
            left = node.get_left()
            right = node.get_right()
            # distance from node to left child = left.dist ? Actually node.dist is the height difference.
            # In scipy linkage, the distance between clusters is stored at node.dist.
            # The branch length from this node to each child is half of the distance? No, for UPGMA the height is the distance from the root to the cluster.
            # Simpler: the depth of a leaf = sum of all node.dist values along the path.
            # Let's compute recursively: each child inherits parent depth + branch length to child.
            # The distance from parent to child is child.dist? Actually the linkage matrix gives the distance between the two clusters that were merged.
            # We'll use the concept of "cophenetic distance" but we want the sum of branch lengths.
            # We'll compute the total branch length from root to leaf as the "height" of the leaf (the y-coordinate in dendrogram).
            # In scipy's to_tree, each node has a 'dist' attribute: the distance between its two child clusters.
            # The root has dist = 0? Actually the last linkage row gives the distance between the two final clusters.
            # The height of a leaf is the sum of all distances from leaf up to root.
            # We'll implement a simple iterative method: for each leaf, climb up.
        pass
    # Alternative: use the linkage matrix directly: find the height at which each leaf merges.
    # Better: compute "cophenetic distance" from root to leaf as the sum of merge distances along path.
    # We'll implement a bottom-up climbing using the linkage matrix.
    # For each leaf, we find the node index, then track parent until root.
    n_leaves = n
    n_clusters = len(linkage_matrix) + 1
    # Build parent dictionary: for each cluster (from 0 to n_clusters-1), parent and branch length to parent.
    parent = [{} for _ in range(n_clusters)]
    # linkage_matrix rows: index 0..n-2, each row: [idx1, idx2, dist, n_elements]
    for i, row in enumerate(linkage_matrix):
        cluster_id = n_leaves + i
        left = int(row[0])
        right = int(row[1])
        dist = row[2]
        parent[left] = {'parent': cluster_id, 'branch_len': dist}
        parent[right] = {'parent': cluster_id, 'branch_len': dist}
    root = n_clusters - 1
    depths = []
    for leaf in range(n_leaves):
        depth = 0.0
        cur = leaf
        while cur != root:
            depth += parent[cur]['branch_len']
            cur = parent[cur]['parent']
        depths.append(depth)
    return depths

def compute_coalescent_scores(returns, metric="correlation", linkage_method="average"):
    """
    Build tree using UPGMA (average linkage) from distance matrix,
    compute root-to-leaf depths for each ETF.
    Returns dict: ticker -> depth (score).
    """
    # Drop any rows with NaN
    returns_clean = returns.dropna()
    if returns_clean.shape[0] < returns_clean.shape[1]:
        # Not enough time points to build stable distances; fallback to zeros
        tickers = returns_clean.columns
        return {t: 0.0 for t in tickers}
    condensed_dist = distance_matrix(returns_clean, metric=metric)
    # Linkage using UPGMA (average)
    Z = linkage(condensed_dist, method=linkage_method, metric='precomputed')
    depths = leaf_depths_from_linkage(Z, returns_clean.shape[1])
    tickers = returns_clean.columns
    return {ticker: depths[i] for i, ticker in enumerate(tickers)}
