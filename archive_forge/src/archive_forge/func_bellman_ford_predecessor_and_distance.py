from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors
@nx._dispatch(edge_attrs='weight')
def bellman_ford_predecessor_and_distance(G, source, target=None, weight='weight', heuristic=False):
    """Compute shortest path lengths and predecessors on shortest paths
    in weighted graphs.

    The algorithm has a running time of $O(mn)$ where $n$ is the number of
    nodes and $m$ is the number of edges.  It is slower than Dijkstra but
    can handle negative edge weights.

    If a negative cycle is detected, you can use :func:`find_negative_cycle`
    to return the cycle and examine it. Shortest paths are not defined when
    a negative cycle exists because once reached, the path can cycle forever
    to build up arbitrarily low weights.

    Parameters
    ----------
    G : NetworkX graph
        The algorithm works for all types of graphs, including directed
        graphs and multigraphs.

    source: node label
        Starting node for path

    target : node label, optional
        Ending node for path

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    heuristic : bool
        Determines whether to use a heuristic to early detect negative
        cycles at a hopefully negligible cost.

    Returns
    -------
    pred, dist : dictionaries
        Returns two dictionaries keyed by node to predecessor in the
        path and to the distance from the source respectively.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXUnbounded
        If the (di)graph contains a negative (di)cycle, the
        algorithm raises an exception to indicate the presence of the
        negative (di)cycle.  Note: any negative weight edge in an
        undirected graph is a negative cycle.

    Examples
    --------
    >>> G = nx.path_graph(5, create_using=nx.DiGraph())
    >>> pred, dist = nx.bellman_ford_predecessor_and_distance(G, 0)
    >>> sorted(pred.items())
    [(0, []), (1, [0]), (2, [1]), (3, [2]), (4, [3])]
    >>> sorted(dist.items())
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    >>> pred, dist = nx.bellman_ford_predecessor_and_distance(G, 0, 1)
    >>> sorted(pred.items())
    [(0, []), (1, [0]), (2, [1]), (3, [2]), (4, [3])]
    >>> sorted(dist.items())
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    >>> G = nx.cycle_graph(5, create_using=nx.DiGraph())
    >>> G[1][2]["weight"] = -7
    >>> nx.bellman_ford_predecessor_and_distance(G, 0)
    Traceback (most recent call last):
        ...
    networkx.exception.NetworkXUnbounded: Negative cycle detected.

    See Also
    --------
    find_negative_cycle

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The dictionaries returned only have keys for nodes reachable from
    the source.

    In the case where the (di)graph is not connected, if a component
    not containing the source contains a negative (di)cycle, it
    will not be detected.

    In NetworkX v2.1 and prior, the source node had predecessor `[None]`.
    In NetworkX v2.2 this changed to the source node having predecessor `[]`
    """
    if source not in G:
        raise nx.NodeNotFound(f'Node {source} is not found in the graph')
    weight = _weight_function(G, weight)
    if G.is_multigraph():
        if any((weight(u, v, {k: d}) < 0 for u, v, k, d in nx.selfloop_edges(G, keys=True, data=True))):
            raise nx.NetworkXUnbounded('Negative cycle detected.')
    elif any((weight(u, v, d) < 0 for u, v, d in nx.selfloop_edges(G, data=True))):
        raise nx.NetworkXUnbounded('Negative cycle detected.')
    dist = {source: 0}
    pred = {source: []}
    if len(G) == 1:
        return (pred, dist)
    weight = _weight_function(G, weight)
    dist = _bellman_ford(G, [source], weight, pred=pred, dist=dist, target=target, heuristic=heuristic)
    return (pred, dist)