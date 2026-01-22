import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_to_pandas_edgelist_with_nodelist():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3)], weight=2.0)
    G.add_edge(0, 5, weight=100)
    df = nx.to_pandas_edgelist(G, nodelist=[1, 2])
    assert 0 not in df['source'].to_numpy()
    assert 100 not in df['weight'].to_numpy()