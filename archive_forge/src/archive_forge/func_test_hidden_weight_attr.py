import pytest
import networkx as nx
def test_hidden_weight_attr():
    G = nx.cycle_graph(3)
    G.add_edge(1, 2, weight=5)
    num_walks = nx.number_of_walks(G, 3)
    expected = {0: {0: 2, 1: 3, 2: 3}, 1: {0: 3, 1: 2, 2: 3}, 2: {0: 3, 1: 3, 2: 2}}
    assert num_walks == expected