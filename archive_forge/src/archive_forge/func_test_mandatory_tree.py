from itertools import product
import pytest
import networkx as nx
def test_mandatory_tree():
    not_a_tree = nx.complete_graph(4)
    with pytest.raises(nx.NotATree):
        nx.community.lukes_partitioning(not_a_tree, 5)