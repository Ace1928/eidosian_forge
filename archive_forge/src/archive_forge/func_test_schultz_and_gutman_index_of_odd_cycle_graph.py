import networkx as nx
def test_schultz_and_gutman_index_of_odd_cycle_graph():
    k = 5
    n = 2 * k + 1
    ocg = nx.cycle_graph(n)
    expected_1 = 2 * n * k * (k + 1)
    actual_1 = nx.schultz_index(ocg)
    expected_2 = 2 * n * k * (k + 1)
    actual_2 = nx.gutman_index(ocg)
    assert expected_1 == actual_1
    assert expected_2 == actual_2