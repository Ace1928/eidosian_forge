import tempfile
from io import BytesIO
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_no_header(self):
    G = nx.complete_graph(4)
    result = BytesIO()
    nx.write_sparse6(G, result, header=False)
    assert result.getvalue() == b':CcKI\n'