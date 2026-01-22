import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_remove_node(self):
    """Tests that removing a node in the original graph
        removes the nodes of the subgraph.

        """
    self.G.remove_node(0)
    assert [1, 3, 4] == sorted(self.H.nodes)
    self.G.add_node(0, name='node0')
    self.G.add_edge(0, 1, name='edge01')