import tempfile
from io import BytesIO
import pytest
import networkx as nx
import networkx.readwrite.graph6 as g6
from networkx.utils import edges_equal, nodes_equal
@pytest.mark.parametrize('G', (nx.MultiGraph(), nx.DiGraph()))
def test_no_directed_or_multi_graphs(self, G):
    with pytest.raises(nx.NetworkXNotImplemented):
        g6.to_graph6_bytes(G)