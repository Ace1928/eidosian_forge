from itertools import cycle, islice
import pytest
import networkx as nx
def test_disconnected_graph_root_node(self):
    """Test for a single component of a disconnected graph."""
    G = nx.barbell_graph(3, 0)
    H = nx.barbell_graph(3, 0)
    mapping = dict(zip(range(6), 'abcdef'))
    nx.relabel_nodes(H, mapping, copy=False)
    G = nx.union(G, H)
    chains = list(nx.chain_decomposition(G, root='a'))
    expected = [[('a', 'b'), ('b', 'c'), ('c', 'a')], [('d', 'e'), ('e', 'f'), ('f', 'd')]]
    assert len(chains) == len(expected)
    for chain in chains:
        self.assertContainsChain(chain, expected)