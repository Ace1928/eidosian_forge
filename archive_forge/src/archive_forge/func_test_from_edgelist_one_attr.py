import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_from_edgelist_one_attr(self):
    Gtrue = nx.Graph([('E', 'C', {'weight': 10}), ('B', 'A', {'weight': 7}), ('A', 'D', {'weight': 4})])
    G = nx.from_pandas_edgelist(self.df, 0, 'b', 'weight')
    assert graphs_equal(G, Gtrue)