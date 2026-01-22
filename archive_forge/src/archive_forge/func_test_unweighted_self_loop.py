import networkx as nx
from networkx.algorithms.approximation import min_weighted_vertex_cover
def test_unweighted_self_loop(self):
    slg = nx.Graph()
    slg.add_node(0)
    slg.add_node(1)
    slg.add_node(2)
    slg.add_edge(0, 1)
    slg.add_edge(2, 2)
    cover = min_weighted_vertex_cover(slg)
    assert 2 == len(cover)
    assert is_cover(slg, cover)