import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_to_edgelist_edge_key_col_exists(self):
    G = nx.path_graph(10, create_using=nx.MultiGraph)
    G.add_weighted_edges_from(((u, v, u) for u, v in list(G.edges())))
    nx.set_edge_attributes(G, 0, name='edge_key_name')
    pytest.raises(nx.NetworkXError, nx.to_pandas_edgelist, G, edge_key='edge_key_name')