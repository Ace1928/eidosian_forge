import networkx as nx
from networkx import is_strongly_regular
def test_is_distance_regular(self):
    assert nx.is_distance_regular(nx.icosahedral_graph())
    assert nx.is_distance_regular(nx.petersen_graph())
    assert nx.is_distance_regular(nx.cubical_graph())
    assert nx.is_distance_regular(nx.complete_bipartite_graph(3, 3))
    assert nx.is_distance_regular(nx.tetrahedral_graph())
    assert nx.is_distance_regular(nx.dodecahedral_graph())
    assert nx.is_distance_regular(nx.pappus_graph())
    assert nx.is_distance_regular(nx.heawood_graph())
    assert nx.is_distance_regular(nx.cycle_graph(3))
    assert not nx.is_distance_regular(nx.path_graph(4))