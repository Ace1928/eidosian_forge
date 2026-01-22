import networkx as nx
from networkx.utils import not_implemented_for
@nx.utils.not_implemented_for('directed')
@nx._dispatch(edge_attrs='weight')
def kemeny_constant(G, *, weight=None):
    """Returns the Kemeny constant of the given graph.

    The *Kemeny constant* (or Kemeny's constant) of a graph `G`
    can be computed by regarding the graph as a Markov chain.
    The Kemeny constant is then the expected number of time steps
    to transition from a starting state i to a random destination state
    sampled from the Markov chain's stationary distribution.
    The Kemeny constant is independent of the chosen initial state [1]_.

    The Kemeny constant measures the time needed for spreading
    across a graph. Low values indicate a closely connected graph
    whereas high values indicate a spread-out graph.

    If weight is not provided, then a weight of 1 is used for all edges.

    Since `G` represents a Markov chain, the weights must be positive.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or None, optional (default=None)
       The edge data key used to compute the Kemeny constant.
       If None, then each edge has weight 1.

    Returns
    -------
    K : float
        The Kemeny constant of the graph `G`.

    Raises
    ------
    NetworkXNotImplemented
        If the graph `G` is directed.

    NetworkXError
        If the graph `G` is not connected, or contains no nodes,
        or has edges with negative weights.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> round(nx.kemeny_constant(G), 10)
    3.2

    Notes
    -----
    The implementation is based on equation (3.3) in [2]_.
    Self-loops are allowed and indicate a Markov chain where
    the state can remain the same. Multi-edges are contracted
    in one edge with weight equal to the sum of the weights.

    References
    ----------
    .. [1] Wikipedia
       "Kemeny's constant."
       https://en.wikipedia.org/wiki/Kemeny%27s_constant
    .. [2] Lovász L.
        Random walks on graphs: A survey.
        Paul Erdös is Eighty, vol. 2, Bolyai Society,
        Mathematical Studies, Keszthely, Hungary (1993), pp. 1-46
    """
    import numpy as np
    import scipy as sp
    if len(G) == 0:
        raise nx.NetworkXError('Graph G must contain at least one node.')
    if not nx.is_connected(G):
        raise nx.NetworkXError('Graph G must be connected.')
    if nx.is_negatively_weighted(G, weight=weight):
        raise nx.NetworkXError('The weights of graph G must be nonnegative.')
    A = nx.adjacency_matrix(G, weight=weight)
    n, m = A.shape
    diags = A.sum(axis=1)
    with np.errstate(divide='ignore'):
        diags_sqrt = 1.0 / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = sp.sparse.csr_array(sp.sparse.spdiags(diags_sqrt, 0, m, n, format='csr'))
    H = DH @ (A @ DH)
    eig = np.sort(sp.linalg.eigvalsh(H.todense()))
    return np.sum(1 / (1 - eig[:-1]))