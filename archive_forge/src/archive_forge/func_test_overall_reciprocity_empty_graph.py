import pytest
import networkx as nx
def test_overall_reciprocity_empty_graph(self):
    with pytest.raises(nx.NetworkXError):
        DG = nx.DiGraph()
        nx.overall_reciprocity(DG)