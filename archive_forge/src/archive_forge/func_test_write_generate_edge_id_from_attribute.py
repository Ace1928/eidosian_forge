import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_write_generate_edge_id_from_attribute(self):
    from xml.etree.ElementTree import parse
    G = nx.Graph()
    G.add_edges_from([('a', 'b'), ('b', 'c'), ('a', 'c')])
    edge_attributes = {e: str(e) for e in G.edges}
    nx.set_edge_attributes(G, edge_attributes, 'eid')
    fd, fname = tempfile.mkstemp()
    self.writer(G, fname, edge_id_from_attribute='eid')
    generator = nx.generate_graphml(G, edge_id_from_attribute='eid')
    H = nx.read_graphml(fname)
    assert nodes_equal(G.nodes(), H.nodes())
    assert edges_equal(G.edges(), H.edges())
    nx.set_edge_attributes(G, edge_attributes, 'id')
    assert edges_equal(G.edges(data=True), H.edges(data=True))
    tree = parse(fname)
    children = list(tree.getroot())
    assert len(children) == 2
    edge_ids = [edge.attrib['id'] for edge in tree.getroot().findall('.//{http://graphml.graphdrawing.org/xmlns}edge')]
    assert sorted(edge_ids) == sorted(edge_attributes.values())
    data = ''.join(generator)
    J = nx.parse_graphml(data)
    assert sorted(G.nodes()) == sorted(J.nodes())
    assert sorted(G.edges()) == sorted(J.edges())
    nx.set_edge_attributes(G, edge_attributes, 'id')
    assert edges_equal(G.edges(data=True), J.edges(data=True))
    os.close(fd)
    os.unlink(fname)