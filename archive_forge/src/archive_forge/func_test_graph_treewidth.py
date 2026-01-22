import pytest
import networkx as nx
def test_graph_treewidth(self):
    with pytest.raises(nx.NetworkXError, match='Input graph is not chordal'):
        nx.chordal_graph_treewidth(self.non_chordal_G)