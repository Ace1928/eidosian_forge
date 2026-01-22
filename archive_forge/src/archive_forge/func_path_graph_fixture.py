from itertools import combinations
import pytest
import networkx as nx
@pytest.fixture(name='path_graph')
def path_graph_fixture():
    return path_graph()