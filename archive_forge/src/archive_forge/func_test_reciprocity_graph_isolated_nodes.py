import pytest
import networkx as nx
def test_reciprocity_graph_isolated_nodes(self):
    with pytest.raises(nx.NetworkXError):
        DG = nx.DiGraph([(1, 2)])
        DG.add_node(4)
        nx.reciprocity(DG, 4)