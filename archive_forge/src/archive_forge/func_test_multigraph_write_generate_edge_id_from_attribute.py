import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_multigraph_write_generate_edge_id_from_attribute(self):
    from xml.etree.ElementTree import parse
    G = nx.MultiGraph()
    G.add_edges_from([('a', 'b'), ('b', 'c'), ('a', 'c'), ('a', 'b')])
    edge_attributes = {e: str(e) for e in G.edges}
    nx.set_edge_attributes(G, edge_attributes, 'eid')
    fd, fname = tempfile.mkstemp()
    self.writer(G, fname, edge_id_from_attribute='eid')
    generator = nx.generate_graphml(G, edge_id_from_attribute='eid')
    H = nx.read_graphml(fname)
    assert H.is_multigraph()
    H = nx.read_graphml(fname, force_multigraph=True)
    assert H.is_multigraph()
    assert nodes_equal(G.nodes(), H.nodes())
    assert edges_equal(G.edges(), H.edges())
    assert sorted((data.get('eid') for u, v, data in H.edges(data=True))) == sorted(edge_attributes.values())
    assert sorted((key for u, v, key in H.edges(keys=True))) == sorted(edge_attributes.values())
    tree = parse(fname)
    children = list(tree.getroot())
    assert len(children) == 2
    edge_ids = [edge.attrib['id'] for edge in tree.getroot().findall('.//{http://graphml.graphdrawing.org/xmlns}edge')]
    assert sorted(edge_ids) == sorted(edge_attributes.values())
    graphml_data = ''.join(generator)
    J = nx.parse_graphml(graphml_data)
    assert J.is_multigraph()
    assert nodes_equal(G.nodes(), J.nodes())
    assert edges_equal(G.edges(), J.edges())
    assert sorted((data.get('eid') for u, v, data in J.edges(data=True))) == sorted(edge_attributes.values())
    assert sorted((key for u, v, key in J.edges(keys=True))) == sorted(edge_attributes.values())
    os.close(fd)
    os.unlink(fname)