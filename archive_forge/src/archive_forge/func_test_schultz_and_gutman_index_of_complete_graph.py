import networkx as nx
def test_schultz_and_gutman_index_of_complete_graph():
    n = 5
    cg = nx.complete_graph(n)
    expected_1 = n * (n - 1) * (n - 1)
    actual_1 = nx.schultz_index(cg)
    assert expected_1 == actual_1
    expected_2 = n * (n - 1) * (n - 1) * (n - 1) / 2
    actual_2 = nx.gutman_index(cg)
    assert expected_2 == actual_2