import one of the named maximum matching algorithms directly.
import collections
import itertools
import networkx as nx
from networkx.algorithms.bipartite import sets as bipartite_sets
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
@nx._dispatch(edge_attrs='weight')
def minimum_weight_full_matching(G, top_nodes=None, weight='weight'):
    """Returns a minimum weight full matching of the bipartite graph `G`.

    Let :math:`G = ((U, V), E)` be a weighted bipartite graph with real weights
    :math:`w : E \\to \\mathbb{R}`. This function then produces a matching
    :math:`M \\subseteq E` with cardinality

    .. math::
       \\lvert M \\rvert = \\min(\\lvert U \\rvert, \\lvert V \\rvert),

    which minimizes the sum of the weights of the edges included in the
    matching, :math:`\\sum_{e \\in M} w(e)`, or raises an error if no such
    matching exists.

    When :math:`\\lvert U \\rvert = \\lvert V \\rvert`, this is commonly
    referred to as a perfect matching; here, since we allow
    :math:`\\lvert U \\rvert` and :math:`\\lvert V \\rvert` to differ, we
    follow Karp [1]_ and refer to the matching as *full*.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    top_nodes : container

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed.

    weight : string, optional (default='weight')

       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    matches : dictionary

      The matching is returned as a dictionary, `matches`, such that
      ``matches[v] == w`` if node `v` is matched to node `w`. Unmatched
      nodes do not occur as a key in `matches`.

    Raises
    ------
    ValueError
      Raised if no full matching exists.

    ImportError
      Raised if SciPy is not available.

    Notes
    -----
    The problem of determining a minimum weight full matching is also known as
    the rectangular linear assignment problem. This implementation defers the
    calculation of the assignment to SciPy.

    References
    ----------
    .. [1] Richard Manning Karp:
       An algorithm to Solve the m x n Assignment Problem in Expected Time
       O(mn log n).
       Networks, 10(2):143–152, 1980.

    """
    import numpy as np
    import scipy as sp
    left, right = nx.bipartite.sets(G, top_nodes)
    U = list(left)
    V = list(right)
    weights_sparse = biadjacency_matrix(G, row_order=U, column_order=V, weight=weight, format='coo')
    weights = np.full(weights_sparse.shape, np.inf)
    weights[weights_sparse.row, weights_sparse.col] = weights_sparse.data
    left_matches = sp.optimize.linear_sum_assignment(weights)
    d = {U[u]: V[v] for u, v in zip(*left_matches)}
    d.update({v: u for u, v in d.items()})
    return d