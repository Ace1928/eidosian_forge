from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@nx._dispatch(graphs=None)
def sedgewick_maze_graph(create_using=None):
    """
    Return a small maze with a cycle.

    This is the maze used in Sedgewick, 3rd Edition, Part 5, Graph
    Algorithms, Chapter 18, e.g. Figure 18.2 and following [1]_.
    Nodes are numbered 0,..,7

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Small maze with a cycle

    References
    ----------
    .. [1] Figure 18.2, Chapter 18, Graph Algorithms (3rd Ed), Sedgewick
    """
    G = empty_graph(0, create_using)
    G.add_nodes_from(range(8))
    G.add_edges_from([[0, 2], [0, 7], [0, 5]])
    G.add_edges_from([[1, 7], [2, 6]])
    G.add_edges_from([[3, 4], [3, 5]])
    G.add_edges_from([[4, 5], [4, 7], [4, 6]])
    G.name = 'Sedgewick Maze'
    return G