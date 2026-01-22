import math
import numbers
from functools import reduce
import networkx as nx
from networkx.utils import nodes_or_number, py_random_state
@py_random_state(3)
@nx._dispatch(graphs=None)
def gnmk_random_graph(n, m, k, seed=None, directed=False):
    """Returns a random bipartite graph G_{n,m,k}.

    Produces a bipartite graph chosen randomly out of the set of all graphs
    with n top nodes, m bottom nodes, and k edges.
    The graph is composed of two sets of nodes.
    Set A has nodes 0 to (n - 1) and set B has nodes n to (n + m - 1).

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set.
    m : int
        The number of nodes in the second bipartite set.
    k : int
        The number of edges
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True return a directed graph

    Examples
    --------
    from nx.algorithms import bipartite
    G = bipartite.gnmk_random_graph(10,20,50)

    See Also
    --------
    gnm_random_graph

    Notes
    -----
    If k > m * n then a complete bipartite graph is returned.

    This graph is a bipartite version of the `G_{nm}` random graph model.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.gnmk_random_graph
    """
    G = nx.Graph()
    G = _add_nodes_with_bipartite_label(G, n, m)
    if directed:
        G = nx.DiGraph(G)
    G.name = f'bipartite_gnm_random_graph({n},{m},{k})'
    if n == 1 or m == 1:
        return G
    max_edges = n * m
    if k >= max_edges:
        return nx.complete_bipartite_graph(n, m, create_using=G)
    top = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
    bottom = list(set(G) - set(top))
    edge_count = 0
    while edge_count < k:
        u = seed.choice(top)
        v = seed.choice(bottom)
        if v in G[u]:
            continue
        else:
            G.add_edge(u, v)
            edge_count += 1
    return G