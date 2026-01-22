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
def test_relabel_duplicate(self):
    data = '\ngraph\n[\n        label   ""\n        directed        1\n        node\n        [\n                id      0\n                label   "same"\n        ]\n        node\n        [\n                id      1\n                label   "same"\n        ]\n]\n'
    fh = io.BytesIO(data.encode('UTF-8'))
    fh.seek(0)
    pytest.raises(nx.NetworkXError, nx.read_gml, fh, label='label')