import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_union_all_and_compose_all():
    K3 = nx.complete_graph(3)
    P3 = nx.path_graph(3)
    G1 = nx.DiGraph()
    G1.add_edge('A', 'B')
    G1.add_edge('A', 'C')
    G1.add_edge('A', 'D')
    G2 = nx.DiGraph()
    G2.add_edge('1', '2')
    G2.add_edge('1', '3')
    G2.add_edge('1', '4')
    G = nx.union_all([G1, G2])
    H = nx.compose_all([G1, G2])
    assert edges_equal(G.edges(), H.edges())
    assert not G.has_edge('A', '1')
    pytest.raises(nx.NetworkXError, nx.union, K3, P3)
    H1 = nx.union_all([H, G1], rename=('H', 'G1'))
    assert sorted(H1.nodes()) == ['G1A', 'G1B', 'G1C', 'G1D', 'H1', 'H2', 'H3', 'H4', 'HA', 'HB', 'HC', 'HD']
    H2 = nx.union_all([H, G2], rename=('H', ''))
    assert sorted(H2.nodes()) == ['1', '2', '3', '4', 'H1', 'H2', 'H3', 'H4', 'HA', 'HB', 'HC', 'HD']
    assert not H1.has_edge('NB', 'NA')
    G = nx.compose_all([G, G])
    assert edges_equal(G.edges(), H.edges())
    G2 = nx.union_all([G2, G2], rename=('', 'copy'))
    assert sorted(G2.nodes()) == ['1', '2', '3', '4', 'copy1', 'copy2', 'copy3', 'copy4']
    assert sorted(G2.neighbors('copy4')) == []
    assert sorted(G2.neighbors('copy1')) == ['copy2', 'copy3', 'copy4']
    assert len(G) == 8
    assert nx.number_of_edges(G) == 6
    E = nx.disjoint_union_all([G, G])
    assert len(E) == 16
    assert nx.number_of_edges(E) == 12
    E = nx.disjoint_union_all([G1, G2])
    assert sorted(E.nodes()) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    G1 = nx.DiGraph()
    G1.add_edge('A', 'B')
    G2 = nx.DiGraph()
    G2.add_edge(1, 2)
    G3 = nx.DiGraph()
    G3.add_edge(11, 22)
    G4 = nx.union_all([G1, G2, G3], rename=('G1', 'G2', 'G3'))
    assert sorted(G4.nodes()) == ['G1A', 'G1B', 'G21', 'G22', 'G311', 'G322']