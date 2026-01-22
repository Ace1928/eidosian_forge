import random
import pytest
import networkx as nx
def test_random_seed():
    G = nx.empty_graph(5)
    assert nx.maximal_independent_set(G, seed=1) == [1, 0, 3, 2, 4]