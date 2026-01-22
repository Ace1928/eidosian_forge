import pytest
import networkx as nx
def test_watts_strogatz_big_k(self):
    pytest.raises(nx.NetworkXError, nx.watts_strogatz_graph, 10, 11, 0.25)
    pytest.raises(nx.NetworkXError, nx.newman_watts_strogatz_graph, 10, 11, 0.25)
    nx.watts_strogatz_graph(10, 9, 0.25, seed=0)
    nx.newman_watts_strogatz_graph(10, 9, 0.5, seed=0)
    nx.watts_strogatz_graph(10, 10, 0.25, seed=0)
    nx.newman_watts_strogatz_graph(10, 10, 0.25, seed=0)