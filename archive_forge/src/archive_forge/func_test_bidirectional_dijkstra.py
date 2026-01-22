import pytest
import networkx as nx
from networkx.utils import pairwise
def test_bidirectional_dijkstra(self):
    validate_length_path(self.XG, 's', 'v', 9, *nx.bidirectional_dijkstra(self.XG, 's', 'v'))
    validate_length_path(self.G, 's', 'v', 2, *nx.bidirectional_dijkstra(self.G, 's', 'v'))
    validate_length_path(self.cycle, 0, 3, 3, *nx.bidirectional_dijkstra(self.cycle, 0, 3))
    validate_length_path(self.cycle, 0, 4, 3, *nx.bidirectional_dijkstra(self.cycle, 0, 4))
    validate_length_path(self.XG3, 0, 3, 15, *nx.bidirectional_dijkstra(self.XG3, 0, 3))
    validate_length_path(self.XG4, 0, 2, 4, *nx.bidirectional_dijkstra(self.XG4, 0, 2))
    P = nx.single_source_dijkstra_path(self.XG, 's')['v']
    validate_path(self.XG, 's', 'v', sum((self.XG[u][v]['weight'] for u, v in zip(P[:-1], P[1:]))), nx.dijkstra_path(self.XG, 's', 'v'))
    G = nx.path_graph(2)
    pytest.raises(nx.NodeNotFound, nx.bidirectional_dijkstra, G, 3, 0)