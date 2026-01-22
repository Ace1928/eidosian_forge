from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_all_pairs_lca_self_ancestors():
    """Self-ancestors should always be the node itself, i.e. lca of (0, 0) is 0.
    See gh-4458."""
    G = nx.DiGraph()
    G.add_nodes_from(range(5))
    G.add_edges_from([(1, 0), (2, 0), (3, 2), (4, 1), (4, 3)])
    ap_lca = nx.all_pairs_lowest_common_ancestor
    assert all((u == v == a for (u, v), a in ap_lca(G) if u == v))
    MG = nx.MultiDiGraph(G)
    assert all((u == v == a for (u, v), a in ap_lca(MG) if u == v))
    MG.add_edges_from([(1, 0), (2, 0)])
    assert all((u == v == a for (u, v), a in ap_lca(MG) if u == v))