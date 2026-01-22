from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@_raise_on_directed
@nx._dispatch(graphs=None)
def diamond_graph(create_using=None):
    """
    Returns the Diamond graph

    The Diamond Graph is  planar undirected graph with 4 nodes and 5 edges.
    It is also sometimes known as the double triangle graph or kite graph [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Diamond Graph with 4 nodes and 5 edges

    References
    ----------
    .. [1] https://mathworld.wolfram.com/DiamondGraph.html
    """
    G = nx.from_dict_of_lists({0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2]}, create_using=create_using)
    G.name = 'Diamond Graph'
    return G