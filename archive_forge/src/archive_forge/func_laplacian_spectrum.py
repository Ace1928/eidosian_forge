import networkx as nx
@nx._dispatch(edge_attrs='weight')
def laplacian_spectrum(G, weight='weight'):
    """Returns eigenvalues of the Laplacian of G

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
    See :func:`~networkx.convert_matrix.to_numpy_array` for other options.

    See Also
    --------
    laplacian_matrix

    Examples
    --------
    The multiplicity of 0 as an eigenvalue of the laplacian matrix is equal
    to the number of connected components of G.

    >>> G = nx.Graph()  # Create a graph with 5 nodes and 3 connected components
    >>> G.add_nodes_from(range(5))
    >>> G.add_edges_from([(0, 2), (3, 4)])
    >>> nx.laplacian_spectrum(G)
    array([0., 0., 0., 2., 2.])

    """
    import scipy as sp
    return sp.linalg.eigvalsh(nx.laplacian_matrix(G, weight=weight).todense())