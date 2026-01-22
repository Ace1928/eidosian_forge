import warnings
import networkx as nx
@nx._dispatch
def single_target_shortest_path_length(G, target, cutoff=None):
    """Compute the shortest path lengths to target from all reachable nodes.

    Parameters
    ----------
    G : NetworkX graph

    target : node
       Target node for path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    lengths : iterator
        (source, shortest path length) iterator

    Examples
    --------
    >>> G = nx.path_graph(5, create_using=nx.DiGraph())
    >>> length = dict(nx.single_target_shortest_path_length(G, 4))
    >>> length[0]
    4
    >>> for node in range(5):
    ...     print(f"{node}: {length[node]}")
    0: 4
    1: 3
    2: 2
    3: 1
    4: 0

    See Also
    --------
    single_source_shortest_path_length, shortest_path_length
    """
    if target not in G:
        raise nx.NodeNotFound(f'Target {target} is not in G')
    msg = 'single_target_shortest_path_length will return a dict starting in v3.3'
    warnings.warn(msg, DeprecationWarning)
    if cutoff is None:
        cutoff = float('inf')
    adj = G._pred if G.is_directed() else G._adj
    nextlevel = [target]
    return _single_shortest_path_length(adj, nextlevel, cutoff)