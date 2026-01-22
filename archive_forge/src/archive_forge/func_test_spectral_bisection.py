from math import sqrt
import pytest
import networkx as nx
def test_spectral_bisection():
    pytest.importorskip('scipy')
    G = nx.barbell_graph(3, 0)
    C = nx.spectral_bisection(G)
    assert C == ({0, 1, 2}, {3, 4, 5})
    mapping = dict(enumerate('badfec'))
    G = nx.relabel_nodes(G, mapping)
    C = nx.spectral_bisection(G)
    assert C == ({mapping[0], mapping[1], mapping[2]}, {mapping[3], mapping[4], mapping[5]})