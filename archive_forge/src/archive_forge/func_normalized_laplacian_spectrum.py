import networkx as nx
@nx._dispatch(edge_attrs='weight')
def normalized_laplacian_spectrum(G, weight='weight'):
    """Return eigenvalues of the normalized Laplacian of G

    Parameters
    ----------
    G : graph
       A NetworkX graph

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    Notes
    -----
    For MultiGraph/MultiDiGraph, the edges weights are summed.
    See to_numpy_array for other options.

    See Also
    --------
    normalized_laplacian_matrix
    """
    import scipy as sp
    return sp.linalg.eigvalsh(nx.normalized_laplacian_matrix(G, weight=weight).todense())