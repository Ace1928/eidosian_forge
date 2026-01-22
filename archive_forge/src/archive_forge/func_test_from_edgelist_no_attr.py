import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_from_edgelist_no_attr(self):
    Gtrue = nx.Graph([('E', 'C', {}), ('B', 'A', {}), ('A', 'D', {})])
    G = nx.from_pandas_edgelist(self.df, 0, 'b')
    assert graphs_equal(G, Gtrue)