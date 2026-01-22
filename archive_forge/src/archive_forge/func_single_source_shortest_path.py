import warnings
import networkx as nx
@nx._dispatch
def single_source_shortest_path(G, source, cutoff=None):
    """Compute shortest path between source
    and all other nodes reachable from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    paths : dictionary
        Dictionary, keyed by target, of shortest paths.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = nx.single_source_shortest_path(G, 0)
    >>> path[4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    The shortest path is not necessarily unique. So there can be multiple
    paths between the source and each target node, all of which have the
    same 'shortest' length. For each target node, this function returns
    only one of those paths.

    See Also
    --------
    shortest_path
    """
    if source not in G:
        raise nx.NodeNotFound(f'Source {source} not in G')

    def join(p1, p2):
        return p1 + p2
    if cutoff is None:
        cutoff = float('inf')
    nextlevel = {source: 1}
    paths = {source: [source]}
    return dict(_single_shortest_path(G.adj, nextlevel, paths, cutoff, join))