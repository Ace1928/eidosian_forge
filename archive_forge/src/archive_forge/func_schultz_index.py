import itertools as it
import networkx as nx
@nx.utils.not_implemented_for('directed')
@nx.utils.not_implemented_for('multigraph')
@nx._dispatchable(edge_attrs='weight')
def schultz_index(G, weight=None):
    """Returns the Schultz Index (of the first kind) of `G`

    The *Schultz Index* [3]_ of a graph is the sum over all node pairs of
    distances times the sum of degrees. Consider an undirected graph `G`.
    For each node pair ``(u, v)`` compute ``dist(u, v) * (deg(u) + deg(v)``
    where ``dist`` is the shortest path length between two nodes and ``deg``
    is the degree of a node.

    The Schultz Index is the sum of these quantities over all (unordered)
    pairs of nodes.

    Parameters
    ----------
    G : NetworkX graph
        The undirected graph of interest.
    weight : string or None, optional (default: None)
        If None, every edge has weight 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        The edge weights are used to computing shortest-path distances.

    Returns
    -------
    number
        The first kind of Schultz Index of the graph `G`.

    Examples
    --------
    The Schultz Index of the (unweighted) complete graph on *n* nodes
    equals the number of pairs of the *n* nodes times ``2 * (n - 1)``,
    since each pair of nodes is at distance one and the sum of degree
    of two nodes is ``2 * (n - 1)``.

    >>> n = 10
    >>> G = nx.complete_graph(n)
    >>> nx.schultz_index(G) == (n * (n - 1) / 2) * (2 * (n - 1))
    True

    Graph that is disconnected

    >>> nx.schultz_index(nx.empty_graph(2))
    inf

    References
    ----------
    .. [1] I. Gutman, Selected properties of the Schultz molecular topological index,
           J. Chem. Inf. Comput. Sci. 34 (1994), 1087–1089.
           https://doi.org/10.1021/ci00021a009
    .. [2] M.V. Diudeaa and I. Gutman, Wiener-Type Topological Indices,
           Croatica Chemica Acta, 71 (1998), 21-51.
           https://hrcak.srce.hr/132323
    .. [3] H. P. Schultz, Topological organic chemistry. 1.
           Graph theory and topological indices of alkanes,i
           J. Chem. Inf. Comput. Sci. 29 (1989), 239–257.

    """
    if not nx.is_connected(G):
        return float('inf')
    spl = nx.shortest_path_length(G, weight=weight)
    d = dict(G.degree, weight=weight)
    return sum((dist * (d[u] + d[v]) for u, info in spl for v, dist in info.items())) / 2