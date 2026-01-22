from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def restricted_view(G, nodes, edges):
    """Returns a view of `G` with hidden nodes and edges.

    The resulting subgraph filters out node `nodes` and edges `edges`.
    Filtered out nodes also filter out any of their edges.

    Parameters
    ----------
    G : NetworkX Graph
    nodes : iterable
        An iterable of nodes. Nodes not present in `G` are ignored.
    edges : iterable
        An iterable of edges. Edges not present in `G` are ignored.

    Returns
    -------
    subgraph : SubGraph View
        A read-only restricted view of `G` filtering out nodes and edges.
        Changes to `G` are reflected in the view.

    Notes
    -----
    To create a mutable subgraph with its own copies of nodes
    edges and attributes use `subgraph.copy()` or `Graph(subgraph)`

    If you create a subgraph of a subgraph recursively you may end up
    with a chain of subgraph views. Such chains can get quite slow
    for lengths near 15. To avoid long chains, try to make your subgraph
    based on the original graph.  We do not rule out chains programmatically
    so that odd cases like an `edge_subgraph` of a `restricted_view`
    can be created.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> H = nx.restricted_view(G, [0], [(1, 2), (3, 4)])
    >>> list(H.nodes)
    [1, 2, 3, 4]
    >>> list(H.edges)
    [(2, 3)]
    """
    nxf = nx.filters
    hide_nodes = nxf.hide_nodes(nodes)
    if G.is_multigraph():
        if G.is_directed():
            hide_edges = nxf.hide_multidiedges(edges)
        else:
            hide_edges = nxf.hide_multiedges(edges)
    elif G.is_directed():
        hide_edges = nxf.hide_diedges(edges)
    else:
        hide_edges = nxf.hide_edges(edges)
    return nx.subgraph_view(G, filter_node=hide_nodes, filter_edge=hide_edges)