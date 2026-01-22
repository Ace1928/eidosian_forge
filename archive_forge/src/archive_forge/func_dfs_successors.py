from collections import defaultdict
import networkx as nx
@nx._dispatch
def dfs_successors(G, source=None, depth_limit=None):
    """Returns dictionary of successors in depth-first-search from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
       Specify starting node for depth-first search.
       Note that you will get successors for all nodes in the
       component containing `source`. This input only specifies
       where the DFS starts.

    depth_limit : int, optional (default=len(G))
       Specify the maximum search depth.

    Returns
    -------
    succ: dict
       A dictionary with nodes as keys and list of successor nodes as values.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.dfs_successors(G, source=0)
    {0: [1], 1: [2], 2: [3], 3: [4]}
    >>> nx.dfs_successors(G, source=0, depth_limit=2)
    {0: [1], 1: [2]}

    Notes
    -----
    If a source is not specified then a source is chosen arbitrarily and
    repeatedly until all components in the graph are searched.

    The implementation of this function is adapted from David Eppstein's
    depth-first search function in `PADS`_, with modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited search`_".

    .. _PADS: http://www.ics.uci.edu/~eppstein/PADS
    .. _Depth-limited search: https://en.wikipedia.org/wiki/Depth-limited_search

    See Also
    --------
    dfs_preorder_nodes
    dfs_postorder_nodes
    dfs_labeled_edges
    edge_dfs
    bfs_tree
    """
    d = defaultdict(list)
    for s, t in dfs_edges(G, source=source, depth_limit=depth_limit):
        d[s].append(t)
    return dict(d)