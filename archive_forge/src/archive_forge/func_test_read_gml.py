import codecs
import io
import math
import os
import tempfile
from ast import literal_eval
from contextlib import contextmanager
from textwrap import dedent
import pytest
import networkx as nx
from networkx.readwrite.gml import literal_destringizer, literal_stringizer
def test_read_gml(self):
    fd, fname = tempfile.mkstemp()
    fh = open(fname, 'w')
    fh.write(self.simple_data)
    fh.close()
    Gin = nx.read_gml(fname, label='label')
    G = nx.parse_gml(self.simple_data, label='label')
    assert sorted(G.nodes(data=True)) == sorted(Gin.nodes(data=True))
    assert sorted(G.edges(data=True)) == sorted(Gin.edges(data=True))
    os.close(fd)
    os.unlink(fname)