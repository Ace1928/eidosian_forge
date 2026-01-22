from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@_raise_on_directed
@nx._dispatch(graphs=None)
def truncated_cube_graph(create_using=None):
    """
    Returns the skeleton of the truncated cube.

    The truncated cube is an Archimedean solid with 14 regular
    faces (6 octagonal and 8 triangular), 36 edges and 24 nodes [1]_.
    The truncated cube is created by truncating (cutting off) the tips
    of the cube one third of the way into each edge [2]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Skeleton of the truncated cube

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Truncated_cube
    .. [2] https://www.coolmath.com/reference/polyhedra-truncated-cube

    """
    G = nx.from_dict_of_lists({0: [1, 2, 4], 1: [11, 14], 2: [3, 4], 3: [6, 8], 4: [5], 5: [16, 18], 6: [7, 8], 7: [10, 12], 8: [9], 9: [17, 20], 10: [11, 12], 11: [14], 12: [13], 13: [21, 22], 14: [15], 15: [19, 23], 16: [17, 18], 17: [20], 18: [19], 19: [23], 20: [21], 21: [22], 22: [23]}, create_using=create_using)
    G.name = 'Truncated Cube Graph'
    return G