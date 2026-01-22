import pytest
import networkx as nx
def test_null_tree2(self):
    with pytest.raises(nx.NetworkXPointlessConcept):
        nx.is_tree(self.multigraph())