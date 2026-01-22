from itertools import combinations
import pytest
import networkx as nx
@pytest.fixture()
def chain_and_fork_graph():
    edge_list = [('A', 'B'), ('B', 'C'), ('B', 'D'), ('D', 'C')]
    G = nx.DiGraph(edge_list)
    return G