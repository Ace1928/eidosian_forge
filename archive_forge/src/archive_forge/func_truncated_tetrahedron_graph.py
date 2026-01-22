from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@nx._dispatch(graphs=None)
def truncated_tetrahedron_graph(create_using=None):
    """
    Returns the skeleton of the truncated Platonic tetrahedron.

    The truncated tetrahedron is an Archimedean solid with 4 regular hexagonal faces,
    4 equilateral triangle faces, 12 nodes and 18 edges. It can be constructed by truncating
    all 4 vertices of a regular tetrahedron at one third of the original edge length [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Skeleton of the truncated tetrahedron

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Truncated_tetrahedron

    """
    G = path_graph(12, create_using)
    G.add_edges_from([(0, 2), (0, 9), (1, 6), (3, 11), (4, 11), (5, 7), (8, 10)])
    G.name = 'Truncated Tetrahedron Graph'
    return G