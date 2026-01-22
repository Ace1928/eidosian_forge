from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('multigraph')
@not_implemented_for('directed')
@nx._dispatch(edge_attrs='weight')
def local_bridges(G, with_span=True, weight=None):
    """Iterate over local bridges of `G` optionally computing the span

    A *local bridge* is an edge whose endpoints have no common neighbors.
    That is, the edge is not part of a triangle in the graph.

    The *span* of a *local bridge* is the shortest path length between
    the endpoints if the local bridge is removed.

    Parameters
    ----------
    G : undirected graph

    with_span : bool
        If True, yield a 3-tuple `(u, v, span)`

    weight : function, string or None (default: None)
        If function, used to compute edge weights for the span.
        If string, the edge data attribute used in calculating span.
        If None, all edges have weight 1.

    Yields
    ------
    e : edge
        The local bridges as an edge 2-tuple of nodes `(u, v)` or
        as a 3-tuple `(u, v, span)` when `with_span is True`.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a directed graph or multigraph.

    Examples
    --------
    A cycle graph has every edge a local bridge with span N-1.

       >>> G = nx.cycle_graph(9)
       >>> (0, 8, 8) in set(nx.local_bridges(G))
       True
    """
    if with_span is not True:
        for u, v in G.edges:
            if not set(G[u]) & set(G[v]):
                yield (u, v)
    else:
        wt = nx.weighted._weight_function(G, weight)
        for u, v in G.edges:
            if not set(G[u]) & set(G[v]):
                enodes = {u, v}

                def hide_edge(n, nbr, d):
                    if n not in enodes or nbr not in enodes:
                        return wt(n, nbr, d)
                    return None
                try:
                    span = nx.shortest_path_length(G, u, v, weight=hide_edge)
                    yield (u, v, span)
                except nx.NetworkXNoPath:
                    yield (u, v, float('inf'))