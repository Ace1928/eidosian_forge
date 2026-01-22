import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='weight')
def modularity_matrix(G, nodelist=None, weight=None):
    """Returns the modularity matrix of G.

    The modularity matrix is the matrix B = A - <A>, where A is the adjacency
    matrix and <A> is the average adjacency matrix, assuming that the graph
    is described by the configuration model.

    More specifically, the element B_ij of B is defined as

    .. math::
        A_{ij} - {k_i k_j \\over 2 m}

    where k_i is the degree of node i, and where m is the number of edges
    in the graph. When weight is set to a name of an attribute edge, Aij, k_i,
    k_j and m are computed using its value.

    Parameters
    ----------
    G : Graph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used for
       the edge weight.  If None then all edge weights are 1.

    Returns
    -------
    B : Numpy array
      The modularity matrix of G.

    Examples
    --------
    >>> k = [3, 2, 2, 1, 0]
    >>> G = nx.havel_hakimi_graph(k)
    >>> B = nx.modularity_matrix(G)


    See Also
    --------
    to_numpy_array
    modularity_spectrum
    adjacency_matrix
    directed_modularity_matrix

    References
    ----------
    .. [1] M. E. J. Newman, "Modularity and community structure in networks",
           Proc. Natl. Acad. Sci. USA, vol. 103, pp. 8577-8582, 2006.
    """
    import numpy as np
    if nodelist is None:
        nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, format='csr')
    k = A.sum(axis=1)
    m = k.sum() * 0.5
    X = np.outer(k, k) / (2 * m)
    return A - X