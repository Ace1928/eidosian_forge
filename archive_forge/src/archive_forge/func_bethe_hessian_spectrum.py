import networkx as nx
@nx._dispatch
def bethe_hessian_spectrum(G, r=None):
    """Returns eigenvalues of the Bethe Hessian matrix of G.

    Parameters
    ----------
    G : Graph
       A NetworkX Graph or DiGraph

    r : float
       Regularizer parameter

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    See Also
    --------
    bethe_hessian_matrix

    References
    ----------
    .. [1] A. Saade, F. Krzakala and L. Zdeborová
       "Spectral clustering of graphs with the bethe hessian",
       Advances in Neural Information Processing Systems. 2014.
    """
    import scipy as sp
    return sp.linalg.eigvalsh(nx.bethe_hessian_matrix(G, r).todense())