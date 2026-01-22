import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_add_half_edge(self):
    embedding = nx.PlanarEmbedding()
    embedding.add_half_edge(0, 1)
    with pytest.raises(nx.NetworkXException, match='Invalid clockwise reference node.'):
        embedding.add_half_edge(0, 2, cw=3)
    with pytest.raises(nx.NetworkXException, match='Invalid counterclockwise reference node.'):
        embedding.add_half_edge(0, 2, ccw=3)
    with pytest.raises(nx.NetworkXException, match='Only one of cw/ccw can be specified.'):
        embedding.add_half_edge(0, 2, cw=1, ccw=1)
    with pytest.raises(nx.NetworkXException, match='Node already has out-half-edge\\(s\\), either cw or ccw reference node required.'):
        embedding.add_half_edge(0, 2)
    embedding.add_half_edge(0, 2, cw=1)
    embedding.add_half_edge(0, 3, ccw=1)
    assert sorted(embedding.edges(data=True)) == [(0, 1, {'ccw': 2, 'cw': 3}), (0, 2, {'cw': 1, 'ccw': 3}), (0, 3, {'cw': 2, 'ccw': 1})]