import itertools
import math
import networkx as nx
from networkx.utils import py_random_state
@nx._dispatch(graphs=None)
def windmill_graph(n, k):
    """Generate a windmill graph.
    A windmill graph is a graph of `n` cliques each of size `k` that are all
    joined at one node.
    It can be thought of as taking a disjoint union of `n` cliques of size `k`,
    selecting one point from each, and contracting all of the selected points.
    Alternatively, one could generate `n` cliques of size `k-1` and one node
    that is connected to all other nodes in the graph.

    Parameters
    ----------
    n : int
        Number of cliques
    k : int
        Size of cliques

    Returns
    -------
    G : NetworkX Graph
        windmill graph with n cliques of size k

    Raises
    ------
    NetworkXError
        If the number of cliques is less than two
        If the size of the cliques are less than two

    Examples
    --------
    >>> G = nx.windmill_graph(4, 5)

    Notes
    -----
    The node labeled `0` will be the node connected to all other nodes.
    Note that windmill graphs are usually denoted `Wd(k,n)`, so the parameters
    are in the opposite order as the parameters of this method.
    """
    if n < 2:
        msg = 'A windmill graph must have at least two cliques'
        raise nx.NetworkXError(msg)
    if k < 2:
        raise nx.NetworkXError('The cliques must have at least two nodes')
    G = nx.disjoint_union_all(itertools.chain([nx.complete_graph(k)], (nx.complete_graph(k - 1) for _ in range(n - 1))))
    G.add_edges_from(((0, i) for i in range(k, G.number_of_nodes())))
    return G