import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def is_k_edge_connected(G, k):
    """Tests to see if a graph is k-edge-connected.

    Is it impossible to disconnect the graph by removing fewer than k edges?
    If so, then G is k-edge-connected.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    k : integer
        edge connectivity to test for

    Returns
    -------
    boolean
        True if G is k-edge-connected.

    See Also
    --------
    :func:`is_locally_k_edge_connected`

    Examples
    --------
    >>> G = nx.barbell_graph(10, 0)
    >>> nx.is_k_edge_connected(G, k=1)
    True
    >>> nx.is_k_edge_connected(G, k=2)
    False
    """
    if k < 1:
        raise ValueError(f'k must be positive, not {k}')
    if G.number_of_nodes() < k + 1:
        return False
    elif any((d < k for n, d in G.degree())):
        return False
    elif k == 1:
        return nx.is_connected(G)
    elif k == 2:
        return nx.is_connected(G) and (not nx.has_bridges(G))
    else:
        return nx.edge_connectivity(G, cutoff=k) >= k