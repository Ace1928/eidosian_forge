import pytest
import networkx
import networkx as nx
from .historical_tests import HistoricalTests
def test_successors(self):
    G = self.G()
    G.add_nodes_from('GJK')
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'C'), ('C', 'D')])
    assert sorted(G.successors('A')) == ['B', 'C']
    assert sorted(G.successors('A')) == ['B', 'C']
    assert sorted(G.successors('G')) == []
    assert sorted(G.successors('D')) == []
    assert sorted(G.successors('G')) == []
    pytest.raises(nx.NetworkXError, G.successors, 'j')
    pytest.raises(nx.NetworkXError, G.successors, 'j')