import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_from_edgelist_int_attr_name(self):
    Gtrue = nx.Graph([('E', 'C', {0: 'C'}), ('B', 'A', {0: 'B'}), ('A', 'D', {0: 'A'})])
    G = nx.from_pandas_edgelist(self.df, 0, 'b', 0)
    assert graphs_equal(G, Gtrue)