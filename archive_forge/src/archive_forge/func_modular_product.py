from itertools import product
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatchable(graphs=_G_H, preserve_edge_attrs=True, preserve_node_attrs=True, returns_graph=True)
def modular_product(G, H):
    """Returns the Modular product of G and H.

    The modular product of `G` and `H` is the graph $M = G \\nabla H$,
    consisting of the node set $V(M) = V(G) \\times V(H)$ that is the Cartesian
    product of the node sets of `G` and `H`. Further, M contains an edge ((u, v), (x, y)):

    - if u is adjacent to x in `G` and v is adjacent to y in `H`, or
    - if u is not adjacent to x in `G` and v is not adjacent to y in `H`.

    More formally::

        E(M) = {((u, v), (x, y)) | ((u, x) in E(G) and (v, y) in E(H)) or
                                   ((u, x) not in E(G) and (v, y) not in E(H))}

    Parameters
    ----------
    G, H: NetworkX graphs
        The graphs to take the modular product of.

    Returns
    -------
    M: NetworkX graph
        The Modular product of `G` and `H`.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is not a simple graph.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> H = nx.path_graph(2)
    >>> M = nx.modular_product(G, H)
    >>> list(M)
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    >>> print(M)
    Graph with 8 nodes and 8 edges

    Notes
    -----
    The *modular product* is defined in [1]_ and was first
    introduced as the *weak modular product*.

    The modular product reduces the problem of counting isomorphic subgraphs
    in `G` and `H` to the problem of counting cliques in M. The subgraphs of
    `G` and `H` that are induced by the nodes of a clique in M are
    isomorphic [2]_ [3]_.

    References
    ----------
    .. [1] R. Hammack, W. Imrich, and S. Klavžar,
        "Handbook of Product Graphs", CRC Press, 2011.

    .. [2] H. G. Barrow and R. M. Burstall,
        "Subgraph isomorphism, matching relational structures and maximal
        cliques", Information Processing Letters, vol. 4, issue 4, pp. 83-84,
        1976, https://doi.org/10.1016/0020-0190(76)90049-1.

    .. [3] V. G. Vizing, "Reduction of the problem of isomorphism and isomorphic
        entrance to the task of finding the nondensity of a graph." Proc. Third
        All-Union Conference on Problems of Theoretical Cybernetics. 1974.
    """
    if G.is_directed() or H.is_directed():
        raise nx.NetworkXNotImplemented('Modular product not implemented for directed graphs')
    if G.is_multigraph() or H.is_multigraph():
        raise nx.NetworkXNotImplemented('Modular product not implemented for multigraphs')
    GH = _init_product_graph(G, H)
    GH.add_nodes_from(_node_product(G, H))
    for u, v, c in G.edges(data=True):
        for x, y, d in H.edges(data=True):
            GH.add_edge((u, x), (v, y), **_dict_product(c, d))
            GH.add_edge((v, x), (u, y), **_dict_product(c, d))
    G = nx.complement(G)
    H = nx.complement(H)
    for u, v, c in G.edges(data=True):
        for x, y, d in H.edges(data=True):
            GH.add_edge((u, x), (v, y), **_dict_product(c, d))
            GH.add_edge((v, x), (u, y), **_dict_product(c, d))
    return GH