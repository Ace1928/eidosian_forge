import networkx as nx
def test_mycielski_graph_generator(self):
    G = nx.mycielski_graph(1)
    assert nx.is_isomorphic(G, nx.empty_graph(1))
    G = nx.mycielski_graph(2)
    assert nx.is_isomorphic(G, nx.path_graph(2))
    G = nx.mycielski_graph(3)
    assert nx.is_isomorphic(G, nx.cycle_graph(5))
    G = nx.mycielski_graph(4)
    assert nx.is_isomorphic(G, nx.mycielskian(nx.cycle_graph(5)))