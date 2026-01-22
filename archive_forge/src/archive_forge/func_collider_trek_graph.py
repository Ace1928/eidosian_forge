from itertools import combinations
import pytest
import networkx as nx
@pytest.fixture()
def collider_trek_graph():
    edge_list = [('A', 'B'), ('C', 'B'), ('C', 'D')]
    G = nx.DiGraph(edge_list)
    return G