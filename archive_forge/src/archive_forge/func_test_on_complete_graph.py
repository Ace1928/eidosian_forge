import collections
import pytest
import networkx as nx
def test_on_complete_graph(self):
    G = nx.complete_graph(4)
    assert nx.is_eulerian(nx.eulerize(G))
    assert nx.is_eulerian(nx.eulerize(nx.MultiGraph(G)))