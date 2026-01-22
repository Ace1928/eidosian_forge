from collections import defaultdict
import networkx as nx
@nx._dispatch
def pad_graph(G, num_colors):
    """Add a disconnected complete clique K_p such that the number of nodes in
    the graph becomes a multiple of `num_colors`.

    Assumes that the graph's nodes are labelled using integers.

    Returns the number of nodes with each color.
    """
    n_ = len(G)
    r = num_colors - 1
    s = n_ // (r + 1)
    if n_ != s * (r + 1):
        p = r + 1 - n_ % (r + 1)
        s += 1
        K = nx.relabel_nodes(nx.complete_graph(p), {idx: idx + n_ for idx in range(p)})
        G.add_edges_from(K.edges)
    return s