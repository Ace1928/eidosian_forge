import itertools
import numbers
import networkx as nx
from networkx.classes import Graph
from networkx.exception import NetworkXError
from networkx.utils import nodes_or_number, pairwise
@nx._dispatch(graphs=None)
def binomial_tree(n, create_using=None):
    """Returns the Binomial Tree of order n.

    The binomial tree of order 0 consists of a single node. A binomial tree of order k
    is defined recursively by linking two binomial trees of order k-1: the root of one is
    the leftmost child of the root of the other.

    Parameters
    ----------
    n : int
        Order of the binomial tree.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : NetworkX graph
        A binomial tree of $2^n$ nodes and $2^n - 1$ edges.

    """
    G = nx.empty_graph(1, create_using)
    N = 1
    for i in range(n):
        edges = [(u + N, v + N) for u, v in G.edges()]
        G.add_edges_from(edges)
        G.add_edge(0, N)
        N *= 2
    return G