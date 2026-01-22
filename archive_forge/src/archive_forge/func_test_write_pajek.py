import os
import tempfile
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_write_pajek(self):
    import io
    G = nx.parse_pajek(self.data)
    fh = io.BytesIO()
    nx.write_pajek(G, fh)
    fh.seek(0)
    H = nx.read_pajek(fh)
    assert nodes_equal(list(G), list(H))
    assert edges_equal(list(G.edges()), list(H.edges()))