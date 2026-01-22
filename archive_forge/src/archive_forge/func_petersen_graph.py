from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@_raise_on_directed
@nx._dispatch(graphs=None)
def petersen_graph(create_using=None):
    """
    Returns the Petersen graph.

    The Peterson graph is a cubic, undirected graph with 10 nodes and 15 edges [1]_.
    Julius Petersen constructed the graph as the smallest counterexample
    against the claim that a connected bridgeless cubic graph
    has an edge colouring with three colours [2]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Petersen graph

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Petersen_graph
    .. [2] https://www.win.tue.nl/~aeb/drg/graphs/Petersen.html
    """
    G = nx.from_dict_of_lists({0: [1, 4, 5], 1: [0, 2, 6], 2: [1, 3, 7], 3: [2, 4, 8], 4: [3, 0, 9], 5: [0, 7, 8], 6: [1, 8, 9], 7: [2, 5, 9], 8: [3, 5, 6], 9: [4, 6, 7]}, create_using=create_using)
    G.name = 'Petersen Graph'
    return G