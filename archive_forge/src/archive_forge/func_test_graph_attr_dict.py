import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_graph_attr_dict(self):
    """Tests that the graph attribute dictionary of the two graphs
        is the same object.

        """
    assert self.G.graph is self.H.graph