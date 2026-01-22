import numpy as np
from scipy import sparse, stats
from scipy.sparse import linalg
from pygsp import graphs, filters, utils
def graph_sparsify(M, epsilon, maxiter=10):
    """Sparsify a graph (with Spielman-Srivastava).

    Parameters
    ----------
    M : Graph or sparse matrix
        Graph structure or a Laplacian matrix
    epsilon : int
        Sparsification parameter

    Returns
    -------
    Mnew : Graph or sparse matrix
        New graph structure or sparse matrix

    Notes
    -----
    Epsilon should be between 1/sqrt(N) and 1

    Examples
    --------
    >>> from pygsp import reduction
    >>> G = graphs.Sensor(256, Nc=20, distribute=True)
    >>> epsilon = 0.4
    >>> G2 = reduction.graph_sparsify(G, epsilon)

    References
    ----------
    See :cite:`spielman2011graph`, :cite:`rudelson1999random` and :cite:`rudelson2007sampling`.
    for more informations

    """
    if isinstance(M, graphs.Graph):
        if not M.lap_type == 'combinatorial':
            raise NotImplementedError
        L = M.L
    else:
        L = M
    N = np.shape(L)[0]
    if not 1.0 / np.sqrt(N) <= epsilon < 1:
        raise ValueError('GRAPH_SPARSIFY: Epsilon out of required range')
    resistance_distances = utils.resistance_distance(L).toarray()
    if isinstance(M, graphs.Graph):
        W = M.W
    else:
        W = np.diag(L.diagonal()) - L.toarray()
        W[W < 1e-10] = 0
    W = sparse.coo_matrix(W)
    W.data[W.data < 1e-10] = 0
    W = W.tocsc()
    W.eliminate_zeros()
    start_nodes, end_nodes, weights = sparse.find(sparse.tril(W))
    weights = np.maximum(0, weights)
    Re = np.maximum(0, resistance_distances[start_nodes, end_nodes])
    Pe = weights * Re
    Pe = Pe / np.sum(Pe)
    for i in range(maxiter):
        C0 = 1 / 30.0
        C = 4 * C0
        q = round(N * np.log(N) * 9 * C ** 2 / epsilon ** 2)
        results = stats.rv_discrete(values=(np.arange(np.shape(Pe)[0]), Pe)).rvs(size=int(q))
        spin_counts = stats.itemfreq(results).astype(int)
        per_spin_weights = weights / (q * Pe)
        counts = np.zeros(np.shape(weights)[0])
        counts[spin_counts[:, 0]] = spin_counts[:, 1]
        new_weights = counts * per_spin_weights
        sparserW = sparse.csc_matrix((new_weights, (start_nodes, end_nodes)), shape=(N, N))
        sparserW = sparserW + sparserW.T
        sparserL = sparse.diags(sparserW.diagonal(), 0) - sparserW
        if graphs.Graph(W=sparserW).is_connected():
            break
        elif i == maxiter - 1:
            logger.warning('Despite attempts to reduce epsilon, sparsified graph is disconnected')
        else:
            epsilon -= (epsilon - 1 / np.sqrt(N)) / 2.0
    if isinstance(M, graphs.Graph):
        sparserW = sparse.diags(sparserL.diagonal(), 0) - sparserL
        if not M.is_directed():
            sparserW = (sparserW + sparserW.T) / 2.0
        Mnew = graphs.Graph(W=sparserW)
    else:
        Mnew = sparse.lil_matrix(sparserL)
    return Mnew