import warnings
import networkx as nx
@nx._dispatch
def predecessor(G, source, target=None, cutoff=None, return_seen=None):
    """Returns dict of predecessors for the path from source to all nodes in G.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    target : node label, optional
       Ending node for path. If provided only predecessors between
       source and target are returned

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    return_seen : bool, optional (default=None)
        Whether to return a dictionary, keyed by node, of the level (number of
        hops) to reach the node (as seen during breadth-first-search).

    Returns
    -------
    pred : dictionary
        Dictionary, keyed by node, of predecessors in the shortest path.


    (pred, seen): tuple of dictionaries
        If `return_seen` argument is set to `True`, then a tuple of dictionaries
        is returned. The first element is the dictionary, keyed by node, of
        predecessors in the shortest path. The second element is the dictionary,
        keyed by node, of the level (number of hops) to reach the node (as seen
        during breadth-first-search).

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> list(G)
    [0, 1, 2, 3]
    >>> nx.predecessor(G, 0)
    {0: [], 1: [0], 2: [1], 3: [2]}
    >>> nx.predecessor(G, 0, return_seen=True)
    ({0: [], 1: [0], 2: [1], 3: [2]}, {0: 0, 1: 1, 2: 2, 3: 3})


    """
    if source not in G:
        raise nx.NodeNotFound(f'Source {source} not in G')
    level = 0
    nextlevel = [source]
    seen = {source: level}
    pred = {source: []}
    while nextlevel:
        level = level + 1
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in G[v]:
                if w not in seen:
                    pred[w] = [v]
                    seen[w] = level
                    nextlevel.append(w)
                elif seen[w] == level:
                    pred[w].append(v)
        if cutoff and cutoff <= level:
            break
    if target is not None:
        if return_seen:
            if target not in pred:
                return ([], -1)
            return (pred[target], seen[target])
        else:
            if target not in pred:
                return []
            return pred[target]
    elif return_seen:
        return (pred, seen)
    else:
        return pred