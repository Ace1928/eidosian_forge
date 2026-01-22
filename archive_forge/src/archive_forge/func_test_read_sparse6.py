import tempfile
from io import BytesIO
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_read_sparse6(self):
    data = b':Q___eDcdFcDeFcE`GaJ`IaHbKNbLM'
    G = nx.from_sparse6_bytes(data)
    fh = BytesIO(data)
    Gin = nx.read_sparse6(fh)
    assert nodes_equal(G.nodes(), Gin.nodes())
    assert edges_equal(G.edges(), Gin.edges())