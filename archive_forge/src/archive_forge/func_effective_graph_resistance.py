import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@nx._dispatchable(edge_attrs='weight')
def effective_graph_resistance(G, weight=None, invert_weight=True):
    """Returns the Effective graph resistance of G.

    Also known as the Kirchhoff index.

    The effective graph resistance is defined as the sum
    of the resistance distance of every node pair in G [1]_.

    If weight is not provided, then a weight of 1 is used for all edges.

    The effective graph resistance of a disconnected graph is infinite.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    weight : string or None, optional (default=None)
       The edge data key used to compute the effective graph resistance.
       If None, then each edge has weight 1.

    invert_weight : boolean (default=True)
        Proper calculation of resistance distance requires building the
        Laplacian matrix with the reciprocal of the weight. Not required
        if the weight is already inverted. Weight cannot be zero.

    Returns
    -------
    RG : float
        The effective graph resistance of `G`.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a directed graph.

    NetworkXError
        If `G` does not contain any nodes.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (1, 4), (3, 4), (3, 5), (4, 5)])
    >>> round(nx.effective_graph_resistance(G), 10)
    10.25

    Notes
    -----
    The implementation is based on Theorem 2.2 in [2]_. Self-loops are ignored.
    Multi-edges are contracted in one edge with weight equal to the harmonic sum of the weights.

    References
    ----------
    .. [1] Wolfram
       "Kirchhoff Index."
       https://mathworld.wolfram.com/KirchhoffIndex.html
    .. [2] W. Ellens, F. M. Spieksma, P. Van Mieghem, A. Jamakovic, R. E. Kooij.
        Effective graph resistance.
        Lin. Alg. Appl. 435:2491-2506, 2011.
    """
    import numpy as np
    if len(G) == 0:
        raise nx.NetworkXError('Graph G must contain at least one node.')
    if not nx.is_connected(G):
        return float('inf')
    G = G.copy()
    if invert_weight and weight is not None:
        if G.is_multigraph():
            for u, v, k, d in G.edges(keys=True, data=True):
                d[weight] = 1 / d[weight]
        else:
            for u, v, d in G.edges(data=True):
                d[weight] = 1 / d[weight]
    mu = np.sort(nx.laplacian_spectrum(G, weight=weight))
    return float(np.sum(1 / mu[1:]) * G.number_of_nodes())