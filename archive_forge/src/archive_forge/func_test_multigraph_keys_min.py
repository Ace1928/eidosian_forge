import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_multigraph_keys_min(self):
    """Tests that the minimum spanning edges of a multigraph
        preserves edge keys.
        """
    G = nx.MultiGraph()
    G.add_edge(0, 1, key='a', weight=2)
    G.add_edge(0, 1, key='b', weight=1)
    min_edges = nx.minimum_spanning_edges
    mst_edges = min_edges(G, algorithm=self.algo, data=False)
    assert edges_equal([(0, 1, 'b')], list(mst_edges))