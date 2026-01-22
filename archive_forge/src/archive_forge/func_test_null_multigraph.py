import collections
import pytest
import networkx as nx
def test_null_multigraph(self):
    with pytest.raises(nx.NetworkXPointlessConcept):
        nx.eulerize(nx.MultiGraph())