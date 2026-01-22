import collections
import pytest
import networkx as nx
def test_on_empty_graph(self):
    with pytest.raises(nx.NetworkXError):
        nx.eulerize(nx.empty_graph(3))