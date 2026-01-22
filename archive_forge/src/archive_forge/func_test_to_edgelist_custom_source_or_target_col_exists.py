import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_to_edgelist_custom_source_or_target_col_exists(self):
    G = nx.path_graph(10)
    G.add_weighted_edges_from(((u, v, u) for u, v in list(G.edges)))
    nx.set_edge_attributes(G, 0, name='source_col_name')
    pytest.raises(nx.NetworkXError, nx.to_pandas_edgelist, G, source='source_col_name')
    for u, v, d in G.edges(data=True):
        d.pop('source_col_name', None)
    nx.set_edge_attributes(G, 0, name='target_col_name')
    pytest.raises(nx.NetworkXError, nx.to_pandas_edgelist, G, target='target_col_name')