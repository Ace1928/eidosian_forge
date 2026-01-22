import pytest
import networkx as nx
def test_null_forest(self):
    with pytest.raises(nx.NetworkXPointlessConcept):
        nx.is_forest(self.graph())