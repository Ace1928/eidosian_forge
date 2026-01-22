import networkx as nx
@nx._dispatch(graphs=_G_H, preserve_all_attrs=True)
def compose(G, H):
    """Compose graph G with H by combining nodes and edges into a single graph.

    The node sets and edges sets do not need to be disjoint.

    Composing preserves the attributes of nodes and edges.
    Attribute values from H take precedent over attribute values from G.

    Parameters
    ----------
    G, H : graph
       A NetworkX graph

    Returns
    -------
    C: A new graph with the same type as G

    See Also
    --------
    :func:`~networkx.Graph.update`
    union
    disjoint_union

    Notes
    -----
    It is recommended that G and H be either both directed or both undirected.

    For MultiGraphs, the edges are identified by incident nodes AND edge-key.
    This can cause surprises (i.e., edge `(1, 2)` may or may not be the same
    in two graphs) if you use MultiGraph without keeping track of edge keys.

    If combining the attributes of common nodes is not desired, consider union(),
    which raises an exception for name collisions.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2)])
    >>> H = nx.Graph([(0, 1), (1, 2)])
    >>> R = nx.compose(G, H)
    >>> R.nodes
    NodeView((0, 1, 2))
    >>> R.edges
    EdgeView([(0, 1), (0, 2), (1, 2)])

    By default, the attributes from `H` take precedent over attributes from `G`.
    If you prefer another way of combining attributes, you can update them after the compose operation:

    >>> G = nx.Graph([(0, 1, {'weight': 2.0}), (3, 0, {'weight': 100.0})])
    >>> H = nx.Graph([(0, 1, {'weight': 10.0}), (1, 2, {'weight': -1.0})])
    >>> nx.set_node_attributes(G, {0: 'dark', 1: 'light', 3: 'black'}, name='color')
    >>> nx.set_node_attributes(H, {0: 'green', 1: 'orange', 2: 'yellow'}, name='color')
    >>> GcomposeH = nx.compose(G, H)

    Normally, color attribute values of nodes of GcomposeH come from H. We can workaround this as follows:

    >>> node_data = {n: G.nodes[n]['color'] + " " + H.nodes[n]['color'] for n in G.nodes & H.nodes}
    >>> nx.set_node_attributes(GcomposeH, node_data, 'color')
    >>> print(GcomposeH.nodes[0]['color'])
    dark green

    >>> print(GcomposeH.nodes[3]['color'])
    black

    Similarly, we can update edge attributes after the compose operation in a way we prefer:

    >>> edge_data = {e: G.edges[e]['weight'] * H.edges[e]['weight'] for e in G.edges & H.edges}
    >>> nx.set_edge_attributes(GcomposeH, edge_data, 'weight')
    >>> print(GcomposeH.edges[(0, 1)]['weight'])
    20.0

    >>> print(GcomposeH.edges[(3, 0)]['weight'])
    100.0
    """
    return nx.compose_all([G, H])