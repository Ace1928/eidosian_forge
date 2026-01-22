import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import py_random_state
@py_random_state(2)
@nx._dispatch(graphs=None)
def duplication_divergence_graph(n, p, seed=None):
    """Returns an undirected graph using the duplication-divergence model.

    A graph of `n` nodes is created by duplicating the initial nodes
    and retaining edges incident to the original nodes with a retention
    probability `p`.

    Parameters
    ----------
    n : int
        The desired number of nodes in the graph.
    p : float
        The probability for retaining the edge of the replicated node.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `p` is not a valid probability.
        If `n` is less than 2.

    Notes
    -----
    This algorithm appears in [1].

    This implementation disallows the possibility of generating
    disconnected graphs.

    References
    ----------
    .. [1] I. Ispolatov, P. L. Krapivsky, A. Yuryev,
       "Duplication-divergence model of protein interaction network",
       Phys. Rev. E, 71, 061911, 2005.

    """
    if p > 1 or p < 0:
        msg = f'NetworkXError p={p} is not in [0,1].'
        raise nx.NetworkXError(msg)
    if n < 2:
        msg = 'n must be greater than or equal to 2'
        raise nx.NetworkXError(msg)
    G = nx.Graph()
    G.add_edge(0, 1)
    i = 2
    while i < n:
        random_node = seed.choice(list(G))
        G.add_node(i)
        flag = False
        for nbr in G.neighbors(random_node):
            if seed.random() < p:
                G.add_edge(i, nbr)
                flag = True
        if not flag:
            G.remove_node(i)
        else:
            i += 1
    return G