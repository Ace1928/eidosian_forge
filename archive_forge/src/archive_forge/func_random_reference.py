import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(3)
@nx._dispatch
def random_reference(G, niter=1, connectivity=True, seed=None):
    """Compute a random graph by swapping edges of a given graph.

    Parameters
    ----------
    G : graph
        An undirected graph with 4 or more nodes.

    niter : integer (optional, default=1)
        An edge is rewired approximately `niter` times.

    connectivity : boolean (optional, default=True)
        When True, ensure connectivity for the randomized graph.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : graph
        The randomized graph.

    Raises
    ------
    NetworkXError
        If there are fewer than 4 nodes or 2 edges in `G`

    Notes
    -----
    The implementation is adapted from the algorithm by Maslov and Sneppen
    (2002) [1]_.

    References
    ----------
    .. [1] Maslov, Sergei, and Kim Sneppen.
           "Specificity and stability in topology of protein networks."
           Science 296.5569 (2002): 910-913.
    """
    if len(G) < 4:
        raise nx.NetworkXError('Graph has fewer than four nodes.')
    if len(G.edges) < 2:
        raise nx.NetworkXError('Graph has fewer that 2 edges')
    from networkx.utils import cumulative_distribution, discrete_sequence
    local_conn = nx.connectivity.local_edge_connectivity
    G = G.copy()
    keys, degrees = zip(*G.degree())
    cdf = cumulative_distribution(degrees)
    nnodes = len(G)
    nedges = nx.number_of_edges(G)
    niter = niter * nedges
    ntries = int(nnodes * nedges / (nnodes * (nnodes - 1) / 2))
    swapcount = 0
    for i in range(niter):
        n = 0
        while n < ntries:
            ai, ci = discrete_sequence(2, cdistribution=cdf, seed=seed)
            if ai == ci:
                continue
            a = keys[ai]
            c = keys[ci]
            b = seed.choice(list(G.neighbors(a)))
            d = seed.choice(list(G.neighbors(c)))
            if b in [a, c, d] or d in [a, b, c]:
                continue
            if d not in G[a] and b not in G[c]:
                G.add_edge(a, d)
                G.add_edge(c, b)
                G.remove_edge(a, b)
                G.remove_edge(c, d)
                if connectivity and local_conn(G, a, b) == 0:
                    G.remove_edge(a, d)
                    G.remove_edge(c, b)
                    G.add_edge(a, b)
                    G.add_edge(c, d)
                else:
                    swapcount += 1
                    break
            n += 1
    return G