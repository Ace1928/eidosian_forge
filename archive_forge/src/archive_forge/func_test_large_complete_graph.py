import tempfile
from io import BytesIO
import pytest
import networkx as nx
import networkx.readwrite.graph6 as g6
from networkx.utils import edges_equal, nodes_equal
def test_large_complete_graph(self):
    G = nx.complete_graph(67)
    assert g6.to_graph6_bytes(G, header=False) == b'~?@B' + b'~' * 368 + b'w\n'