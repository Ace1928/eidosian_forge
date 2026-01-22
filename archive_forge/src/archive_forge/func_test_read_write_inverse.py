import tempfile
from io import BytesIO
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_read_write_inverse(self):
    for i in list(range(13)) + [31, 47, 62, 63, 64, 72]:
        m = min(2 * i, i * i // 2)
        g = nx.random_graphs.gnm_random_graph(i, m, seed=i)
        gstr = BytesIO()
        nx.write_sparse6(g, gstr, header=False)
        gstr = gstr.getvalue().rstrip()
        g2 = nx.from_sparse6_bytes(gstr)
        assert g2.order() == g.order()
        assert edges_equal(g2.edges(), g.edges())