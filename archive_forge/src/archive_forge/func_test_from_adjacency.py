import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_from_adjacency(self):
    nodelist = [1, 2]
    dftrue = pd.DataFrame([[1, 1], [1, 0]], dtype=int, index=nodelist, columns=nodelist)
    G = nx.Graph([(1, 1), (1, 2)])
    df = nx.to_pandas_adjacency(G, dtype=int)
    pd.testing.assert_frame_equal(df, dftrue)