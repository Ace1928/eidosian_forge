from itertools import combinations
import pytest
import networkx as nx
@pytest.fixture(name='collider_graph')
def collider_graph_fixture():
    return collider_graph()