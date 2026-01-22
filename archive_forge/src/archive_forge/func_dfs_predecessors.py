from collections import defaultdict
import networkx as nx
@nx._dispatch
def dfs_predecessors(G, source=None, depth_limit=None):
    """Returns dictionary of predecessors in depth-first-search from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
       Specify starting node for depth-first search.
       Note that you will get predecessors for all nodes in the
       component containing `source`. This input only specifies
       where the DFS starts.

    depth_limit : int, optional (default=len(G))
       Specify the maximum search depth.

    Returns
    -------
    pred: dict
       A dictionary with nodes as keys and predecessor nodes as values.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.dfs_predecessors(G, source=0)
    {1: 0, 2: 1, 3: 2}
    >>> nx.dfs_predecessors(G, source=0, depth_limit=2)
    {1: 0, 2: 1}

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
    return {t: s for s, t in dfs_edges(G, source, depth_limit)}