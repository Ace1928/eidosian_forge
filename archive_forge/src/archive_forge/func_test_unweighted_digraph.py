import networkx as nx
def test_unweighted_digraph(self):
    G = nx.DiGraph(nx.cycle_graph(3))
    vitality = nx.closeness_vitality(G)
    assert vitality == {0: 4, 1: 4, 2: 4}