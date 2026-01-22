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
def test_graph_types(self):
    for directed in [None, False, True]:
        for multigraph in [None, False, True]:
            gml = 'graph ['
            if directed is not None:
                gml += ' directed ' + str(int(directed))
            if multigraph is not None:
                gml += ' multigraph ' + str(int(multigraph))
            gml += ' node [ id 0 label "0" ]'
            gml += ' edge [ source 0 target 0 ]'
            gml += ' ]'
            G = nx.parse_gml(gml)
            assert bool(directed) == G.is_directed()
            assert bool(multigraph) == G.is_multigraph()
            gml = 'graph [\n'
            if directed is True:
                gml += '  directed 1\n'
            if multigraph is True:
                gml += '  multigraph 1\n'
            gml += '  node [\n    id 0\n    label "0"\n  ]\n  edge [\n    source 0\n    target 0\n'
            if multigraph:
                gml += '    key 0\n'
            gml += '  ]\n]'
            assert gml == '\n'.join(nx.generate_gml(G))