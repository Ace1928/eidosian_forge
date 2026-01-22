from itertools import chain
import pytest
import networkx as nx
def test_alternating_path(self):
    G = nx.DiGraph(chain.from_iterable(([(i, i - 1), (i, i + 1)] for i in range(0, 100, 2))))
    assert not nx.is_semiconnected(G)