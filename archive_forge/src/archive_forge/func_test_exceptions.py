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
def test_exceptions(self):
    pytest.raises(ValueError, literal_destringizer, '(')
    pytest.raises(ValueError, literal_destringizer, 'frozenset([1, 2, 3])')
    pytest.raises(ValueError, literal_destringizer, literal_destringizer)
    pytest.raises(ValueError, literal_stringizer, frozenset([1, 2, 3]))
    pytest.raises(ValueError, literal_stringizer, literal_stringizer)
    with tempfile.TemporaryFile() as f:
        f.write(codecs.BOM_UTF8 + b'graph[]')
        f.seek(0)
        pytest.raises(nx.NetworkXError, nx.read_gml, f)

    def assert_parse_error(gml):
        pytest.raises(nx.NetworkXError, nx.parse_gml, gml)
    assert_parse_error(['graph [\n\n', ']'])
    assert_parse_error('')
    assert_parse_error('Creator ""')
    assert_parse_error('0')
    assert_parse_error('graph ]')
    assert_parse_error('graph [ 1 ]')
    assert_parse_error('graph [ 1.E+2 ]')
    assert_parse_error('graph [ "A" ]')
    assert_parse_error('graph [ ] graph ]')
    assert_parse_error('graph [ ] graph [ ]')
    assert_parse_error('graph [ data [1, 2, 3] ]')
    assert_parse_error('graph [ node [ ] ]')
    assert_parse_error('graph [ node [ id 0 ] ]')
    nx.parse_gml('graph [ node [ id "a" ] ]', label='id')
    assert_parse_error('graph [ node [ id 0 label 0 ] node [ id 0 label 1 ] ]')
    assert_parse_error('graph [ node [ id 0 label 0 ] node [ id 1 label 0 ] ]')
    assert_parse_error('graph [ node [ id 0 label 0 ] edge [ ] ]')
    assert_parse_error('graph [ node [ id 0 label 0 ] edge [ source 0 ] ]')
    nx.parse_gml('graph [edge [ source 0 target 0 ] node [ id 0 label 0 ] ]')
    assert_parse_error('graph [ node [ id 0 label 0 ] edge [ source 1 target 0 ] ]')
    assert_parse_error('graph [ node [ id 0 label 0 ] edge [ source 0 target 1 ] ]')
    assert_parse_error('graph [ node [ id 0 label 0 ] node [ id 1 label 1 ] edge [ source 0 target 1 ] edge [ source 1 target 0 ] ]')
    nx.parse_gml('graph [ node [ id 0 label 0 ] node [ id 1 label 1 ] edge [ source 0 target 1 ] edge [ source 1 target 0 ] directed 1 ]')
    nx.parse_gml('graph [ node [ id 0 label 0 ] node [ id 1 label 1 ] edge [ source 0 target 1 ] edge [ source 0 target 1 ]multigraph 1 ]')
    nx.parse_gml('graph [ node [ id 0 label 0 ] node [ id 1 label 1 ] edge [ source 0 target 1 key 0 ] edge [ source 0 target 1 ]multigraph 1 ]')
    assert_parse_error('graph [ node [ id 0 label 0 ] node [ id 1 label 1 ] edge [ source 0 target 1 key 0 ] edge [ source 0 target 1 key 0 ]multigraph 1 ]')
    nx.parse_gml('graph [ node [ id 0 label 0 ] node [ id 1 label 1 ] edge [ source 0 target 1 key 0 ] edge [ source 1 target 0 key 0 ]directed 1 multigraph 1 ]')
    nx.parse_gml('graph [edge [ source a target a ] node [ id a label b ] ]')
    nx.parse_gml('graph [ node [ id n42 label 0 ] node [ id x43 label 1 ]edge [ source n42 target x43 key 0 ]edge [ source x43 target n42 key 0 ]directed 1 multigraph 1 ]')
    assert_parse_error("graph [edge [ source u'uĐ0' target u'uĐ0' ] " + "node [ id u'uĐ0' label b ] ]")

    def assert_generate_error(*args, **kwargs):
        pytest.raises(nx.NetworkXError, lambda: list(nx.generate_gml(*args, **kwargs)))
    G = nx.Graph()
    G.graph[3] = 3
    assert_generate_error(G)
    G = nx.Graph()
    G.graph['3'] = 3
    assert_generate_error(G)
    G = nx.Graph()
    G.graph['data'] = frozenset([1, 2, 3])
    assert_generate_error(G, stringizer=literal_stringizer)