from collections import defaultdict
import networkx as nx
@nx._dispatch
def get_counterexample_recursive(G):
    """Recursive version of :meth:`get_counterexample`."""
    G = nx.Graph(G)
    if check_planarity_recursive(G)[0]:
        raise nx.NetworkXException('G is planar - no counter example.')
    subgraph = nx.Graph()
    for u in G:
        nbrs = list(G[u])
        for v in nbrs:
            G.remove_edge(u, v)
            if check_planarity_recursive(G)[0]:
                G.add_edge(u, v)
                subgraph.add_edge(u, v)
    return subgraph