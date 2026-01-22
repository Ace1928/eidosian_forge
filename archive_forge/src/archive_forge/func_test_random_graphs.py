import random
import pytest
import networkx as nx
def test_random_graphs():
    """Generate 5 random graphs of different types and sizes and
    make sure that all sets are independent and maximal."""
    for i in range(0, 50, 10):
        G = nx.erdos_renyi_graph(i * 10 + 1, random.random())
        IS = nx.maximal_independent_set(G)
        assert G.subgraph(IS).number_of_edges() == 0
        neighbors_of_MIS = set.union(*(set(G.neighbors(v)) for v in IS))
        assert all((v in neighbors_of_MIS for v in set(G.nodes()).difference(IS)))