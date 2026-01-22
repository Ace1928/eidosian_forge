import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_edgekey_with_multigraph(self):
    df = pd.DataFrame({'source': {'A': 'N1', 'B': 'N2', 'C': 'N1', 'D': 'N1'}, 'target': {'A': 'N2', 'B': 'N3', 'C': 'N1', 'D': 'N2'}, 'attr1': {'A': 'F1', 'B': 'F2', 'C': 'F3', 'D': 'F4'}, 'attr2': {'A': 1, 'B': 0, 'C': 0, 'D': 0}, 'attr3': {'A': 0, 'B': 1, 'C': 0, 'D': 1}})
    Gtrue = nx.MultiGraph([('N1', 'N2', 'F1', {'attr2': 1, 'attr3': 0}), ('N2', 'N3', 'F2', {'attr2': 0, 'attr3': 1}), ('N1', 'N1', 'F3', {'attr2': 0, 'attr3': 0}), ('N1', 'N2', 'F4', {'attr2': 0, 'attr3': 1})])
    G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=['attr2', 'attr3'], edge_key='attr1', create_using=nx.MultiGraph())
    assert graphs_equal(G, Gtrue)
    df_roundtrip = nx.to_pandas_edgelist(G, edge_key='attr1')
    df_roundtrip = df_roundtrip.sort_values('attr1')
    df_roundtrip.index = ['A', 'B', 'C', 'D']
    pd.testing.assert_frame_equal(df, df_roundtrip[['source', 'target', 'attr1', 'attr2', 'attr3']])