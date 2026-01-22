import itertools
import numbers
import networkx as nx
from networkx.classes import Graph
from networkx.exception import NetworkXError
from networkx.utils import nodes_or_number, pairwise
@nodes_or_number(0)
@nx._dispatch(graphs=None)
def wheel_graph(n, create_using=None):
    """Return the wheel graph

    The wheel graph consists of a hub node connected to a cycle of (n-1) nodes.

    Parameters
    ----------
    n : int or iterable
        If an integer, node labels are 0 to n with center 0.
        If an iterable of nodes, the center is the first.
        Warning: n is not checked for duplicates and if present the
        resulting graph may not be as desired. Make sure you have no duplicates.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Node labels are the integers 0 to n - 1.
    """
    _, nodes = n
    G = empty_graph(nodes, create_using)
    if G.is_directed():
        raise NetworkXError('Directed Graph not supported')
    if len(nodes) > 1:
        hub, *rim = nodes
        G.add_edges_from(((hub, node) for node in rim))
        if len(rim) > 1:
            G.add_edges_from(pairwise(rim, cyclic=True))
    return G