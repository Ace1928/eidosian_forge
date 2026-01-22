from collections import defaultdict
from itertools import combinations, permutations
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@not_implemented_for('undirected')
@py_random_state(1)
@nx._dispatch
def random_triad(G, seed=None):
    """Returns a random triad from a directed graph.

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G2 : subgraph
       A randomly selected triad (order-3 NetworkX DiGraph)

    Raises
    ------
    NetworkXError
        If the input Graph has less than 3 nodes.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3), (3, 1), (5, 6), (5, 4), (6, 7)])
    >>> triad = nx.random_triad(G, seed=1)
    >>> triad.edges
    OutEdgeView([(1, 2)])

    """
    if len(G) < 3:
        raise nx.NetworkXError(f'G needs at least 3 nodes to form a triad; (it has {len(G)} nodes)')
    nodes = seed.sample(list(G.nodes()), 3)
    G2 = G.subgraph(nodes)
    return G2