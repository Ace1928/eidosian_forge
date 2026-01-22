import io
import time
import pytest
import networkx as nx
def test_dynamic_mode(self):
    G = nx.Graph()
    G.add_node(1, label='1', color='green')
    G.graph['mode'] = 'dynamic'
    fh = io.BytesIO()
    nx.write_gexf(G, fh)
    fh.seek(0)
    H = nx.read_gexf(fh, node_type=int)
    assert sorted(G.nodes()) == sorted(H.nodes())
    assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))