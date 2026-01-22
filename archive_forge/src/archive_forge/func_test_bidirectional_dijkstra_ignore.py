import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_bidirectional_dijkstra_ignore():
    G = nx.Graph()
    nx.add_path(G, [1, 2, 10])
    nx.add_path(G, [1, 3, 10])
    pytest.raises(nx.NetworkXNoPath, _bidirectional_dijkstra, G, 1, 2, ignore_nodes=[1])
    pytest.raises(nx.NetworkXNoPath, _bidirectional_dijkstra, G, 1, 2, ignore_nodes=[2])
    pytest.raises(nx.NetworkXNoPath, _bidirectional_dijkstra, G, 1, 2, ignore_nodes=[1, 2])