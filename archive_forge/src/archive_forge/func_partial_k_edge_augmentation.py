import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@nx._dispatch
def partial_k_edge_augmentation(G, k, avail, weight=None):
    """Finds augmentation that k-edge-connects as much of the graph as possible.

    When a k-edge-augmentation is not possible, we can still try to find a
    small set of edges that partially k-edge-connects as much of the graph as
    possible. All possible edges are generated between remaining parts.
    This minimizes the number of k-edge-connected subgraphs in the resulting
    graph and maximizes the edge connectivity between those subgraphs.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    k : integer
        Desired edge connectivity

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    Yields
    ------
    edge : tuple
        Edges in the partial augmentation of G. These edges k-edge-connect any
        part of G where it is possible, and maximally connects the remaining
        parts. In other words, all edges from avail are generated except for
        those within subgraphs that have already become k-edge-connected.

    Notes
    -----
    Construct H that augments G with all edges in avail.
    Find the k-edge-subgraphs of H.
    For each k-edge-subgraph, if the number of nodes is more than k, then find
    the k-edge-augmentation of that graph and add it to the solution. Then add
    all edges in avail between k-edge subgraphs to the solution.

    See Also
    --------
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))
    >>> G.add_node(8)
    >>> avail = [(1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 5), (1, 8)]
    >>> sorted(partial_k_edge_augmentation(G, k=2, avail=avail))
    [(1, 5), (1, 8)]
    """

    def _edges_between_disjoint(H, only1, only2):
        """finds edges between disjoint nodes"""
        only1_adj = {u: set(H.adj[u]) for u in only1}
        for u, neighbs in only1_adj.items():
            neighbs12 = neighbs.intersection(only2)
            for v in neighbs12:
                yield (u, v)
    avail_uv, avail_w = _unpack_available_edges(avail, weight=weight, G=G)
    H = G.copy()
    H.add_edges_from(((u, v, {'weight': w, 'generator': (u, v)}) for (u, v), w in zip(avail, avail_w)))
    k_edge_subgraphs = list(nx.k_edge_subgraphs(H, k=k))
    for nodes in k_edge_subgraphs:
        if len(nodes) > 1:
            C = H.subgraph(nodes).copy()
            sub_avail = {d['generator']: d['weight'] for u, v, d in C.edges(data=True) if 'generator' in d}
            C.remove_edges_from(sub_avail.keys())
            yield from nx.k_edge_augmentation(C, k=k, avail=sub_avail)
    for cc1, cc2 in it.combinations(k_edge_subgraphs, 2):
        for u, v in _edges_between_disjoint(H, cc1, cc2):
            d = H.get_edge_data(u, v)
            edge = d.get('generator', None)
            if edge is not None:
                yield edge