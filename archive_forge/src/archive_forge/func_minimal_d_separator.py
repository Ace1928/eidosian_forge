from collections import deque
import networkx as nx
from networkx.utils import UnionFind, not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch
def minimal_d_separator(G, u, v):
    """Compute a minimal d-separating set between 'u' and 'v'.

    A d-separating set in a DAG is a set of nodes that blocks all paths
    between the two nodes, 'u' and 'v'. This function
    constructs a d-separating set that is "minimal", meaning it is the smallest
    d-separating set for 'u' and 'v'. This is not necessarily
    unique. For more details, see Notes.

    Parameters
    ----------
    G : graph
        A networkx DAG.
    u : node
        A node in the graph, G.
    v : node
        A node in the graph, G.

    Raises
    ------
    NetworkXError
        Raises a :exc:`NetworkXError` if the input graph is not a DAG.

    NodeNotFound
        If any of the input nodes are not found in the graph,
        a :exc:`NodeNotFound` exception is raised.

    References
    ----------
    .. [1] Tian, J., & Paz, A. (1998). Finding Minimal D-separators.

    Notes
    -----
    This function only finds ``a`` minimal d-separator. It does not guarantee
    uniqueness, since in a DAG there may be more than one minimal d-separator
    between two nodes. Moreover, this only checks for minimal separators
    between two nodes, not two sets. Finding minimal d-separators between
    two sets of nodes is not supported.

    Uses the algorithm presented in [1]_. The complexity of the algorithm
    is :math:`O(|E_{An}^m|)`, where :math:`|E_{An}^m|` stands for the
    number of edges in the moralized graph of the sub-graph consisting
    of only the ancestors of 'u' and 'v'. For full details, see [1]_.

    The algorithm works by constructing the moral graph consisting of just
    the ancestors of `u` and `v`. Then it constructs a candidate for
    a separating set  ``Z'`` from the predecessors of `u` and `v`.
    Then BFS is run starting from `u` and marking nodes
    found from ``Z'`` and calling those nodes ``Z''``.
    Then BFS is run again starting from `v` and marking nodes if they are
    present in ``Z''``. Those marked nodes are the returned minimal
    d-separating set.

    https://en.wikipedia.org/wiki/Bayesian_network#d-separation
    """
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError('graph should be directed acyclic')
    union_uv = {u, v}
    if any((n not in G.nodes for n in union_uv)):
        raise nx.NodeNotFound('one or more specified nodes not found in the graph')
    x_anc = nx.ancestors(G, u)
    y_anc = nx.ancestors(G, v)
    D_anc_xy = x_anc.union(y_anc)
    D_anc_xy.update((u, v))
    moral_G = nx.moral_graph(G.subgraph(D_anc_xy))
    Z_prime = set(G.predecessors(u)).union(set(G.predecessors(v)))
    Z_dprime = _bfs_with_marks(moral_G, u, Z_prime)
    Z = _bfs_with_marks(moral_G, v, Z_dprime)
    return Z