import warnings
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
@pytest.mark.parametrize('G', (nx.Graph(), nx.DiGraph(), nx.MultiGraph(), nx.MultiDiGraph()))
def test_agraph_roundtripping(self, G, tmp_path):
    G = self.build_graph(G)
    A = nx.nx_agraph.to_agraph(G)
    H = nx.nx_agraph.from_agraph(A)
    self.assert_equal(G, H)
    fname = tmp_path / 'test.dot'
    nx.drawing.nx_agraph.write_dot(H, fname)
    Hin = nx.nx_agraph.read_dot(fname)
    self.assert_equal(H, Hin)
    fname = tmp_path / 'fh_test.dot'
    with open(fname, 'w') as fh:
        nx.drawing.nx_agraph.write_dot(H, fh)
    with open(fname) as fh:
        Hin = nx.nx_agraph.read_dot(fh)
    self.assert_equal(H, Hin)