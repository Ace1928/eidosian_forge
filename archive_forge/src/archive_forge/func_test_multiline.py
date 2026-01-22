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
def test_multiline(self):
    multiline_example = '\ngraph\n[\n    node\n    [\n\t    id 0\n\t    label "multiline node"\n\t    label2 "multiline1\n    multiline2\n    multiline3"\n\t    alt_name "id 0"\n    ]\n]\n'
    G = nx.parse_gml(multiline_example)
    assert G.nodes['multiline node'] == {'label2': 'multiline1 multiline2 multiline3', 'alt_name': 'id 0'}