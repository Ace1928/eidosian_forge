import networkx as nx
from networkx.algorithms.approximation import (
def is_clique(G, nodes):
    """Returns True if and only if `nodes` is an independent set
    in `G`.

    `G` is an undirected simple graph. `nodes` is an iterable of
    nodes in `G`.

    """
    H = G.subgraph(nodes)
    n = len(H)
    return H.number_of_edges() == n * (n - 1) // 2