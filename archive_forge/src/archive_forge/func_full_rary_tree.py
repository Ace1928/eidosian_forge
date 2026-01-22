import itertools
import numbers
import networkx as nx
from networkx.classes import Graph
from networkx.exception import NetworkXError
from networkx.utils import nodes_or_number, pairwise
@nx._dispatch(graphs=None)
def full_rary_tree(r, n, create_using=None):
    """Creates a full r-ary tree of `n` nodes.

    Sometimes called a k-ary, n-ary, or m-ary tree.
    "... all non-leaf nodes have exactly r children and all levels
    are full except for some rightmost position of the bottom level
    (if a leaf at the bottom level is missing, then so are all of the
    leaves to its right." [1]_

    Parameters
    ----------
    r : int
        branching factor of the tree
    n : int
        Number of nodes in the tree
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        An r-ary tree with n nodes

    References
    ----------
    .. [1] An introduction to data structures and algorithms,
           James Andrew Storer,  Birkhauser Boston 2001, (page 225).
    """
    G = empty_graph(n, create_using)
    G.add_edges_from(_tree_edges(n, r))
    return G