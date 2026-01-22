import math
from collections import deque
import networkx as nx
@nx._dispatch
def bfs_predecessors(G, source, depth_limit=None, sort_neighbors=None):
    """Returns an iterator of predecessors in breadth-first-search from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Specify starting node for breadth-first search

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    sort_neighbors : function
        A function that takes the list of neighbors of given node as input, and
        returns an *iterator* over these neighbors but with custom ordering.

    Returns
    -------
    pred: iterator
        (node, predecessor) iterator where `predecessor` is the predecessor of
        `node` in a breadth first search starting from `source`.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> dict(nx.bfs_predecessors(G, 0))
    {1: 0, 2: 1}
    >>> H = nx.Graph()
    >>> H.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    >>> dict(nx.bfs_predecessors(H, 0))
    {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    >>> M = nx.Graph()
    >>> nx.add_path(M, [0, 1, 2, 3, 4, 5, 6])
    >>> nx.add_path(M, [2, 7, 8, 9, 10])
    >>> sorted(nx.bfs_predecessors(M, source=1, depth_limit=3))
    [(0, 1), (2, 1), (3, 2), (4, 3), (7, 2), (8, 7)]
    >>> N = nx.DiGraph()
    >>> nx.add_path(N, [0, 1, 2, 3, 4, 7])
    >>> nx.add_path(N, [3, 5, 6, 7])
    >>> sorted(nx.bfs_predecessors(N, source=2))
    [(3, 2), (4, 3), (5, 3), (6, 5), (7, 4)]

    Notes
    -----
    Based on http://www.ics.uci.edu/~eppstein/PADS/BFS.py
    by D. Eppstein, July 2004. The modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited-search`_".

    .. _Depth-limited-search: https://en.wikipedia.org/wiki/Depth-limited_search

    See Also
    --------
    bfs_tree
    bfs_edges
    edge_bfs
    """
    for s, t in bfs_edges(G, source, depth_limit=depth_limit, sort_neighbors=sort_neighbors):
        yield (t, s)