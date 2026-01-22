import networkx as nx
Returns eigenvalues of the Bethe Hessian matrix of G.

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
    