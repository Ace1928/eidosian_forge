import networkx as nx
def test_wiener_index_of_disconnected_graph():
    assert nx.wiener_index(nx.empty_graph(2)) == float('inf')