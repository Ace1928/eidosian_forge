from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@nx._dispatch(graphs=None)
def dodecahedral_graph(create_using=None):
    """
    Returns the Platonic Dodecahedral graph.

    The dodecahedral graph has 20 nodes and 30 edges. The skeleton of the
    dodecahedron forms a graph. It is one of 5 Platonic graphs [1]_.
    It can be described in LCF notation as:
    ``[10, 7, 4, -4, -7, 10, -4, 7, -7, 4]^2`` [2]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Dodecahedral Graph with 20 nodes and 30 edges

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Regular_dodecahedron#Dodecahedral_graph
    .. [2] https://mathworld.wolfram.com/DodecahedralGraph.html

    """
    G = LCF_graph(20, [10, 7, 4, -4, -7, 10, -4, 7, -7, 4], 2, create_using)
    G.name = 'Dodecahedral Graph'
    return G