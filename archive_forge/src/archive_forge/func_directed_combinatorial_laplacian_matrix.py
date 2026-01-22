import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('undirected')
@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='weight')
def directed_combinatorial_laplacian_matrix(G, nodelist=None, weight='weight', walk_type=None, alpha=0.95):
    """Return the directed combinatorial Laplacian matrix of G.

    The graph directed combinatorial Laplacian is the matrix

    .. math::

        L = \\Phi - (\\Phi P + P^T \\Phi) / 2

    where `P` is the transition matrix of the graph and `\\Phi` a matrix
    with the Perron vector of `P` in the diagonal and zeros elsewhere [1]_.

    Depending on the value of walk_type, `P` can be the transition matrix
    induced by a random walk, a lazy random walk, or a random walk with
    teleportation (PageRank).

    Parameters
    ----------
    G : DiGraph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    walk_type : string or None, optional (default=None)
       If None, `P` is selected depending on the properties of the
       graph. Otherwise is one of 'random', 'lazy', or 'pagerank'

    alpha : real
       (1 - alpha) is the teleportation probability used with pagerank

    Returns
    -------
    L : NumPy matrix
      Combinatorial Laplacian of G.

    Notes
    -----
    Only implemented for DiGraphs

    See Also
    --------
    laplacian_matrix

    References
    ----------
    .. [1] Fan Chung (2005).
       Laplacians and the Cheeger inequality for directed graphs.
       Annals of Combinatorics, 9(1), 2005
    """
    import scipy as sp
    P = _transition_matrix(G, nodelist=nodelist, weight=weight, walk_type=walk_type, alpha=alpha)
    n, m = P.shape
    evals, evecs = sp.sparse.linalg.eigs(P.T, k=1)
    v = evecs.flatten().real
    p = v / v.sum()
    Phi = sp.sparse.csr_array(sp.sparse.spdiags(p, 0, n, n)).toarray()
    return Phi - (Phi @ P + P.T @ Phi) / 2.0