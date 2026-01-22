import math
from collections import deque
import networkx as nx
@nx._dispatch
def generic_bfs_edges(G, source, neighbors=None, depth_limit=None, sort_neighbors=None):
    """Iterate over edges in a breadth-first search.

    The breadth-first search begins at `source` and enqueues the
    neighbors of newly visited nodes specified by the `neighbors`
    function.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node for the breadth-first search; this function
        iterates over only those edges in the component reachable from
        this node.

    neighbors : function
        A function that takes a newly visited node of the graph as input
        and returns an *iterator* (not just a list) of nodes that are
        neighbors of that node with custom ordering. If not specified, this is
        just the``G.neighbors`` method, but in general it can be any function
        that returns an iterator over some or all of the neighbors of a
        given node, in any order.

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth.

    sort_neighbors : Callable

        .. deprecated:: 3.2

           The sort_neighbors parameter is deprecated and will be removed in
           version 3.4. A custom (e.g. sorted) ordering of neighbors can be
           specified with the `neighbors` parameter.

        A function that takes the list of neighbors of a given node as input,
        and returns an iterator over these neighbors but with a custom
        ordering.

    Yields
    ------
    edge
        Edges in the breadth-first search starting from `source`.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> list(nx.bfs_edges(G, 0))
    [(0, 1), (1, 2)]
    >>> list(nx.bfs_edges(G, source=0, depth_limit=1))
    [(0, 1)]

    Notes
    -----
    This implementation is from `PADS`_, which was in the public domain
    when it was first accessed in July, 2004.  The modifications
    to allow depth limits are based on the Wikipedia article
    "`Depth-limited-search`_".

    .. _PADS: http://www.ics.uci.edu/~eppstein/PADS/BFS.py
    .. _Depth-limited-search: https://en.wikipedia.org/wiki/Depth-limited_search
    """
    if neighbors is None:
        neighbors = G.neighbors
    if sort_neighbors is not None:
        import warnings
        warnings.warn('The sort_neighbors parameter is deprecated and will be removed\nin NetworkX 3.4, use the neighbors parameter instead.', DeprecationWarning, stacklevel=2)
        _neighbors = neighbors
        neighbors = lambda node: iter(sort_neighbors(_neighbors(node)))
    if depth_limit is None:
        depth_limit = len(G)
    seen = {source}
    n = len(G)
    depth = 0
    next_parents_children = [(source, neighbors(source))]
    while next_parents_children and depth < depth_limit:
        this_parents_children = next_parents_children
        next_parents_children = []
        for parent, children in this_parents_children:
            for child in children:
                if child not in seen:
                    seen.add(child)
                    next_parents_children.append((child, neighbors(child)))
                    yield (parent, child)
            if len(seen) == n:
                return
        depth += 1