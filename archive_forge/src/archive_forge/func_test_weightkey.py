import math
from operator import eq
import networkx as nx
import networkx.algorithms.isomorphism as iso
def test_weightkey():
    g1 = nx.DiGraph()
    g2 = nx.DiGraph()
    g1.add_edge('A', 'B', weight=1)
    g2.add_edge('C', 'D', weight=0)
    assert nx.is_isomorphic(g1, g2)
    em = iso.numerical_edge_match('nonexistent attribute', 1)
    assert nx.is_isomorphic(g1, g2, edge_match=em)
    em = iso.numerical_edge_match('weight', 1)
    assert not nx.is_isomorphic(g1, g2, edge_match=em)
    g2 = nx.DiGraph()
    g2.add_edge('C', 'D')
    assert nx.is_isomorphic(g1, g2, edge_match=em)