import pytest
import networkx as nx
def test_reciprocity_graph_node(self):
    DG = nx.DiGraph([(1, 2), (2, 3), (3, 2)])
    reciprocity = nx.reciprocity(DG, 2)
    assert reciprocity == 0.6666666666666666