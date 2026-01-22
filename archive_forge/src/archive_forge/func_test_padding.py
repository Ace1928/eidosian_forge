import tempfile
from io import BytesIO
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_padding(self):
    codes = (b':Cdv', b':DaYn', b':EaYnN', b':FaYnL', b':GaYnLz')
    for n, code in enumerate(codes, start=4):
        G = nx.path_graph(n)
        result = BytesIO()
        nx.write_sparse6(G, result, header=False)
        assert result.getvalue() == code + b'\n'