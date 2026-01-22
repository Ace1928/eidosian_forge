from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def selfloop_edges(G, data=False, keys=False, default=None):
    """Returns an iterator over selfloop edges.

    A selfloop edge has the same node at both ends.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    data : string or bool, optional (default=False)
        Return selfloop edges as two tuples (u, v) (data=False)
        or three-tuples (u, v, datadict) (data=True)
        or three-tuples (u, v, datavalue) (data='attrname')
    keys : bool, optional (default=False)
        If True, return edge keys with each edge.
    default : value, optional (default=None)
        Value used for edges that don't have the requested attribute.
        Only relevant if data is not True or False.

    Returns
    -------
    edgeiter : iterator over edge tuples
        An iterator over all selfloop edges.

    See Also
    --------
    nodes_with_selfloops, number_of_selfloops

    Examples
    --------
    >>> G = nx.MultiGraph()  # or Graph, DiGraph, MultiDiGraph, etc
    >>> ekey = G.add_edge(1, 1)
    >>> ekey = G.add_edge(1, 2)
    >>> list(nx.selfloop_edges(G))
    [(1, 1)]
    >>> list(nx.selfloop_edges(G, data=True))
    [(1, 1, {})]
    >>> list(nx.selfloop_edges(G, keys=True))
    [(1, 1, 0)]
    >>> list(nx.selfloop_edges(G, keys=True, data=True))
    [(1, 1, 0, {})]
    """
    if data is True:
        if G.is_multigraph():
            if keys is True:
                return ((n, n, k, d) for n, nbrs in G.adj.items() if n in nbrs for k, d in nbrs[n].items())
            else:
                return ((n, n, d) for n, nbrs in G.adj.items() if n in nbrs for d in nbrs[n].values())
        else:
            return ((n, n, nbrs[n]) for n, nbrs in G.adj.items() if n in nbrs)
    elif data is not False:
        if G.is_multigraph():
            if keys is True:
                return ((n, n, k, d.get(data, default)) for n, nbrs in G.adj.items() if n in nbrs for k, d in nbrs[n].items())
            else:
                return ((n, n, d.get(data, default)) for n, nbrs in G.adj.items() if n in nbrs for d in nbrs[n].values())
        else:
            return ((n, n, nbrs[n].get(data, default)) for n, nbrs in G.adj.items() if n in nbrs)
    elif G.is_multigraph():
        if keys is True:
            return ((n, n, k) for n, nbrs in G.adj.items() if n in nbrs for k in nbrs[n])
        else:
            return ((n, n) for n, nbrs in G.adj.items() if n in nbrs for i in range(len(nbrs[n])))
    else:
        return ((n, n) for n, nbrs in G.adj.items() if n in nbrs)