import pytest
import networkx as nx
def test_null_forest2(self):
    with pytest.raises(nx.NetworkXPointlessConcept):
        nx.is_forest(self.multigraph())