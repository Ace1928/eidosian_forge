import pytest
import networkx as nx
def test_reciprocity_graph_nodes(self):
    DG = nx.DiGraph([(1, 2), (2, 3), (3, 2)])
    reciprocity = nx.reciprocity(DG, [1, 2])
    expected_reciprocity = {1: 0.0, 2: 0.6666666666666666}
    assert reciprocity == expected_reciprocity