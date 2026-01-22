import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_not_implemented_decorator():

    @not_implemented_for('directed')
    def test_d(G):
        pass
    test_d(nx.Graph())
    with pytest.raises(nx.NetworkXNotImplemented):
        test_d(nx.DiGraph())

    @not_implemented_for('undirected')
    def test_u(G):
        pass
    test_u(nx.DiGraph())
    with pytest.raises(nx.NetworkXNotImplemented):
        test_u(nx.Graph())

    @not_implemented_for('multigraph')
    def test_m(G):
        pass
    test_m(nx.Graph())
    with pytest.raises(nx.NetworkXNotImplemented):
        test_m(nx.MultiGraph())

    @not_implemented_for('graph')
    def test_g(G):
        pass
    test_g(nx.MultiGraph())
    with pytest.raises(nx.NetworkXNotImplemented):
        test_g(nx.Graph())

    @not_implemented_for('directed', 'multigraph')
    def test_not_md(G):
        pass
    test_not_md(nx.Graph())
    test_not_md(nx.DiGraph())
    test_not_md(nx.MultiGraph())
    with pytest.raises(nx.NetworkXNotImplemented):
        test_not_md(nx.MultiDiGraph())

    @not_implemented_for('directed')
    @not_implemented_for('multigraph')
    def test_graph_only(G):
        pass
    test_graph_only(nx.Graph())
    with pytest.raises(nx.NetworkXNotImplemented):
        test_graph_only(nx.DiGraph())
    with pytest.raises(nx.NetworkXNotImplemented):
        test_graph_only(nx.MultiGraph())
    with pytest.raises(nx.NetworkXNotImplemented):
        test_graph_only(nx.MultiDiGraph())
    with pytest.raises(ValueError):
        not_implemented_for('directed', 'undirected')
    with pytest.raises(ValueError):
        not_implemented_for('multigraph', 'graph')