import itertools
import math
import networkx as nx
from networkx.utils import py_random_state
@nx._dispatch(graphs=None)
def connected_caveman_graph(l, k):
    """Returns a connected caveman graph of `l` cliques of size `k`.

    The connected caveman graph is formed by creating `n` cliques of size
    `k`, then a single edge in each clique is rewired to a node in an
    adjacent clique.

    Parameters
    ----------
    l : int
      number of cliques
    k : int
      size of cliques (k at least 2 or NetworkXError is raised)

    Returns
    -------
    G : NetworkX Graph
      connected caveman graph

    Raises
    ------
    NetworkXError
        If the size of cliques `k` is smaller than 2.

    Notes
    -----
    This returns an undirected graph, it can be converted to a directed
    graph using :func:`nx.to_directed`, or a multigraph using
    ``nx.MultiGraph(nx.caveman_graph(l, k))``. Only the undirected version is
    described in [1]_ and it is unclear which of the directed
    generalizations is most useful.

    Examples
    --------
    >>> G = nx.connected_caveman_graph(3, 3)

    References
    ----------
    .. [1] Watts, D. J. 'Networks, Dynamics, and the Small-World Phenomenon.'
       Amer. J. Soc. 105, 493-527, 1999.
    """
    if k < 2:
        raise nx.NetworkXError('The size of cliques in a connected caveman graph must be at least 2.')
    G = nx.caveman_graph(l, k)
    for start in range(0, l * k, k):
        G.remove_edge(start, start + 1)
        G.add_edge(start, (start - 1) % (l * k))
    return G