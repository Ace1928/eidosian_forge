import itertools
import math
import networkx as nx
from networkx.utils import py_random_state
@nx._dispatch(graphs=None)
def ring_of_cliques(num_cliques, clique_size):
    """Defines a "ring of cliques" graph.

    A ring of cliques graph is consisting of cliques, connected through single
    links. Each clique is a complete graph.

    Parameters
    ----------
    num_cliques : int
        Number of cliques
    clique_size : int
        Size of cliques

    Returns
    -------
    G : NetworkX Graph
        ring of cliques graph

    Raises
    ------
    NetworkXError
        If the number of cliques is lower than 2 or
        if the size of cliques is smaller than 2.

    Examples
    --------
    >>> G = nx.ring_of_cliques(8, 4)

    See Also
    --------
    connected_caveman_graph

    Notes
    -----
    The `connected_caveman_graph` graph removes a link from each clique to
    connect it with the next clique. Instead, the `ring_of_cliques` graph
    simply adds the link without removing any link from the cliques.
    """
    if num_cliques < 2:
        raise nx.NetworkXError('A ring of cliques must have at least two cliques')
    if clique_size < 2:
        raise nx.NetworkXError('The cliques must have at least two nodes')
    G = nx.Graph()
    for i in range(num_cliques):
        edges = itertools.combinations(range(i * clique_size, i * clique_size + clique_size), 2)
        G.add_edges_from(edges)
        G.add_edge(i * clique_size + 1, (i + 1) * clique_size % (num_cliques * clique_size))
    return G