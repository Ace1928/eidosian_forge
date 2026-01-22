import networkx as nx
@nx._dispatch(graphs=_G_H)
def symmetric_difference(G, H):
    """Returns new graph with edges that exist in either G or H but not both.

    The node sets of H and G must be the same.

    Parameters
    ----------
    G,H : graph
       A NetworkX graph.  G and H must have the same node sets.

    Returns
    -------
    D : A new graph with the same type as G.

    Notes
    -----
    Attributes from the graph, nodes, and edges are not copied to the new
    graph.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3)])
    >>> H = nx.Graph([(0, 1), (1, 2), (0, 3)])
    >>> R = nx.symmetric_difference(G, H)
    >>> R.nodes
    NodeView((0, 1, 2, 3))
    >>> R.edges
    EdgeView([(0, 2), (0, 3), (1, 3)])
    """
    if not G.is_multigraph() == H.is_multigraph():
        raise nx.NetworkXError('G and H must both be graphs or multigraphs.')
    R = nx.create_empty_copy(G)
    if set(G) != set(H):
        raise nx.NetworkXError('Node sets of graphs not equal')
    gnodes = set(G)
    hnodes = set(H)
    nodes = gnodes.symmetric_difference(hnodes)
    R.add_nodes_from(nodes)
    if G.is_multigraph():
        edges = G.edges(keys=True)
    else:
        edges = G.edges()
    for e in edges:
        if not H.has_edge(*e):
            R.add_edge(*e)
    if H.is_multigraph():
        edges = H.edges(keys=True)
    else:
        edges = H.edges()
    for e in edges:
        if not G.has_edge(*e):
            R.add_edge(*e)
    return R