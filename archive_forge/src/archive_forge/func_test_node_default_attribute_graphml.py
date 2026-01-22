import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_node_default_attribute_graphml(self):
    G = self.node_attribute_default_graph
    H = nx.read_graphml(self.node_attribute_default_fh)
    assert G.graph['node_default'] == H.graph['node_default']