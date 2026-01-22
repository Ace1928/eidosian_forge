import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch(edge_attrs='weight')
def total_spanning_tree_weight(G, weight=None):
    """
    Returns the total weight of all spanning trees of `G`.

    Kirchoff's Tree Matrix Theorem states that the determinant of any cofactor of the
    Laplacian matrix of a graph is the number of spanning trees in the graph. For a
    weighted Laplacian matrix, it is the sum across all spanning trees of the
    multiplicative weight of each tree. That is, the weight of each tree is the
    product of its edge weights.

    Parameters
    ----------
    G : NetworkX Graph
        The graph to use Kirchhoff's theorem on.

    weight : string or None
        The key for the edge attribute holding the edge weight. If `None`, then
        each edge is assumed to have a weight of 1 and this function returns the
        total number of spanning trees in `G`.

    Returns
    -------
    float
        The sum of the total multiplicative weights for all spanning trees in `G`
    """
    import numpy as np
    G_laplacian = nx.laplacian_matrix(G, weight=weight).toarray()
    return abs(np.linalg.det(G_laplacian[1:, 1:]))