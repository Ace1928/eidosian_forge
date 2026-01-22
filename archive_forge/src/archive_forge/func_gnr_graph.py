import numbers
from collections import Counter
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import discrete_sequence, py_random_state, weighted_choice
@py_random_state(3)
@nx._dispatch(graphs=None)
def gnr_graph(n, p, create_using=None, seed=None):
    """Returns the growing network with redirection (GNR) digraph with `n`
    nodes and redirection probability `p`.

    The GNR graph is built by adding nodes one at a time with a link to one
    previously added node.  The previous target node is chosen uniformly at
    random.  With probability `p` the link is instead "redirected" to the
    successor node of the target.

    The graph is always a (directed) tree.

    Parameters
    ----------
    n : int
        The number of nodes for the generated graph.
    p : float
        The redirection probability.
    create_using : NetworkX graph constructor, optional (default DiGraph)
        Graph type to create. If graph instance, then cleared before populated.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Examples
    --------
    To create the undirected GNR graph, use the :meth:`~DiGraph.to_directed`
    method::

    >>> D = nx.gnr_graph(10, 0.5)  # the GNR graph
    >>> G = D.to_undirected()  # the undirected version

    References
    ----------
    .. [1] P. L. Krapivsky and S. Redner,
           Organization of Growing Random Networks,
           Phys. Rev. E, 63, 066123, 2001.
    """
    G = empty_graph(1, create_using, default=nx.DiGraph)
    if not G.is_directed():
        raise nx.NetworkXError('create_using must indicate a Directed Graph')
    if n == 1:
        return G
    for source in range(1, n):
        target = seed.randrange(0, source)
        if seed.random() < p and target != 0:
            target = next(G.successors(target))
        G.add_edge(source, target)
    return G