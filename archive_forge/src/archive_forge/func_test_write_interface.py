import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_write_interface(self):
    try:
        import lxml.etree
        assert nx.write_graphml == nx.write_graphml_lxml
    except ImportError:
        assert nx.write_graphml == nx.write_graphml_xml