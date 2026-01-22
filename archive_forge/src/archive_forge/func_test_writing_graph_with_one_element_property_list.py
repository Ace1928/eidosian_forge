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
def test_writing_graph_with_one_element_property_list(self):
    g = nx.Graph()
    g.add_node('n1', properties=['element'])
    with byte_file() as f:
        nx.write_gml(g, f)
    result = f.read().decode()
    assert result == dedent('            graph [\n              node [\n                id 0\n                label "n1"\n                properties "_networkx_list_start"\n                properties "element"\n              ]\n            ]\n        ')