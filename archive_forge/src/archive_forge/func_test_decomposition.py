from itertools import cycle, islice
import pytest
import networkx as nx
def test_decomposition(self):
    edges = [(1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (1, 3), (1, 4), (2, 5), (5, 10), (6, 8)]
    G = nx.Graph(edges)
    expected = [[(1, 3), (3, 2), (2, 1)], [(1, 4), (4, 3)], [(2, 5), (5, 3)], [(5, 10), (10, 9), (9, 5)], [(6, 8), (8, 7), (7, 6)]]
    chains = list(nx.chain_decomposition(G, root=1))
    assert len(chains) == len(expected)