import networkx as nx
def test_weighted_multidigraph(self):
    G = nx.MultiDiGraph()
    nx.add_cycle(G, [0, 1, 2], weight=2)
    nx.add_cycle(G, [2, 1, 0], weight=2)
    vitality = nx.closeness_vitality(G, weight='weight')
    assert vitality == {0: 8, 1: 8, 2: 8}