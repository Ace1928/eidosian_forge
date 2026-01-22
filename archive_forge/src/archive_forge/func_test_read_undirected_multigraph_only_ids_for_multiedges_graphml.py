import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_read_undirected_multigraph_only_ids_for_multiedges_graphml(self):
    G = self.multigraph_only_ids_for_multiedges
    H = nx.read_graphml(self.multigraph_only_ids_for_multiedges_fh)
    assert nodes_equal(G.nodes(), H.nodes())
    assert edges_equal(G.edges(), H.edges())
    self.multigraph_only_ids_for_multiedges_fh.seek(0)
    PG = nx.parse_graphml(self.multigraph_only_ids_for_multiedges_data)
    assert nodes_equal(G.nodes(), PG.nodes())
    assert edges_equal(G.edges(), PG.edges())