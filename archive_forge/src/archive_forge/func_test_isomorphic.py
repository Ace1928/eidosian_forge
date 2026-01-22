import pytest
import networkx as nx
from networkx.generators import directed
def test_isomorphic():
    """
    graph hashes should be invariant to node-relabeling (when the output is reindexed
    by the same mapping)
    """
    n, r = (100, 10)
    p = 1.0 / r
    for i in range(1, r + 1):
        G1 = nx.erdos_renyi_graph(n, p * i, seed=200 + i)
        G2 = nx.relabel_nodes(G1, {u: -1 * u for u in G1.nodes()})
        g1_hash = nx.weisfeiler_lehman_graph_hash(G1)
        g2_hash = nx.weisfeiler_lehman_graph_hash(G2)
        assert g1_hash == g2_hash