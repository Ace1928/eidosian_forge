import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_latin1_err(self):
    G = nx.Graph()
    name1 = chr(2344) + chr(123) + chr(6543)
    name2 = chr(5543) + chr(1543) + chr(324)
    G.add_edge(name1, 'Radiohead', **{name2: 3})
    fd, fname = tempfile.mkstemp()
    pytest.raises(UnicodeEncodeError, nx.write_multiline_adjlist, G, fname, encoding='latin-1')
    os.close(fd)
    os.unlink(fname)