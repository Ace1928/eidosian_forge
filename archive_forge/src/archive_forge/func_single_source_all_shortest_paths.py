import warnings
import networkx as nx
@nx._dispatch(edge_attrs='weight')
def single_source_all_shortest_paths(G, source, weight=None, method='dijkstra'):
    """Compute all shortest simple paths from the given source in the graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path.

    weight : None, string or function, optional (default = None)
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly
        three positional arguments: the two endpoints of an edge and
        the dictionary of edge attributes for that edge.
        The function must return a number.

    method : string, optional (default = 'dijkstra')
       The algorithm to use to compute the path lengths.
       Supported options: 'dijkstra', 'bellman-ford'.
       Other inputs produce a ValueError.
       If `weight` is None, unweighted graph methods are used, and this
       suggestion is ignored.

    Returns
    -------
    paths : generator of dictionary
        A generator of all paths between source and all nodes in the graph.

    Raises
    ------
    ValueError
        If `method` is not among the supported options.

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_path(G, [0, 1, 2, 3, 0])
    >>> dict(nx.single_source_all_shortest_paths(G, source=0))
    {0: [[0]], 1: [[0, 1]], 2: [[0, 1, 2], [0, 3, 2]], 3: [[0, 3]]}

    Notes
    -----
    There may be many shortest paths between the source and target.  If G
    contains zero-weight cycles, this function will not produce all shortest
    paths because doing so would produce infinitely many paths of unbounded
    length -- instead, we only produce the shortest simple paths.

    See Also
    --------
    shortest_path
    all_shortest_paths
    single_source_shortest_path
    all_pairs_shortest_path
    all_pairs_all_shortest_paths
    """
    method = 'unweighted' if weight is None else method
    if method == 'unweighted':
        pred = nx.predecessor(G, source)
    elif method == 'dijkstra':
        pred, dist = nx.dijkstra_predecessor_and_distance(G, source, weight=weight)
    elif method == 'bellman-ford':
        pred, dist = nx.bellman_ford_predecessor_and_distance(G, source, weight=weight)
    else:
        raise ValueError(f'method not supported: {method}')
    for n in G:
        try:
            yield (n, list(_build_paths_from_predecessors({source}, n, pred)))
        except nx.NetworkXNoPath:
            pass