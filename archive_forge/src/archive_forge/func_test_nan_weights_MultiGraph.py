import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nan_weights_MultiGraph(self):
    G = nx.MultiGraph()
    G.add_edge(0, 12, weight=float('nan'))
    edges = nx.minimum_spanning_edges(G, algorithm='prim', data=False, ignore_nan=False)
    with pytest.raises(ValueError):
        list(edges)
    edges = nx.minimum_spanning_edges(G, algorithm='prim', data=False)
    with pytest.raises(ValueError):
        list(edges)