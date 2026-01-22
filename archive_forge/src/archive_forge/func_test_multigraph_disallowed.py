import pytest
import networkx as nx
def test_multigraph_disallowed(self):
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.stochastic_graph(nx.MultiGraph())