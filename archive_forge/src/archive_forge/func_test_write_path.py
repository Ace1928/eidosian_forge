import tempfile
from io import BytesIO
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_write_path(self):
    with tempfile.NamedTemporaryFile() as f:
        fullfilename = f.name
    nx.write_sparse6(nx.null_graph(), fullfilename)
    fh = open(fullfilename, mode='rb')
    assert fh.read() == b'>>sparse6<<:?\n'
    fh.close()
    import os
    os.remove(fullfilename)