import networkx as nx
from networkx.utils import py_random_state
@py_random_state(3)
@nx._dispatch(graphs=None)
def general_random_intersection_graph(n, m, p, seed=None):
    """Returns a random intersection graph with independent probabilities
    for connections between node and attribute sets.

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set (nodes)
    m : int
        The number of nodes in the second bipartite set (attributes)
    p : list of floats of length m
        Probabilities for connecting nodes to each attribute
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    gnp_random_graph, uniform_random_intersection_graph

    References
    ----------
    .. [1] Nikoletseas, S. E., Raptopoulos, C., and Spirakis, P. G.
       The existence and efficient construction of large independent sets
       in general random intersection graphs. In ICALP (2004), J. D´ıaz,
       J. Karhum¨aki, A. Lepist¨o, and D. Sannella, Eds., vol. 3142
       of Lecture Notes in Computer Science, Springer, pp. 1029–1040.
    """
    if len(p) != m:
        raise ValueError('Probability list p must have m elements.')
    G = nx.empty_graph(n + m)
    mset = range(n, n + m)
    for u in range(n):
        for v, q in zip(mset, p):
            if seed.random() < q:
                G.add_edge(u, v)
    return nx.projected_graph(G, range(n))