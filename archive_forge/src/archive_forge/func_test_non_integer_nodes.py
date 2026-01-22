import pytest
import networkx as nx
def test_non_integer_nodes():
    G = nx.DiGraph([('A', 'B'), ('B', 'C'), ('C', 'A')])
    num_walks = nx.number_of_walks(G, 2)
    expected = {'A': {'A': 0, 'B': 0, 'C': 1}, 'B': {'A': 1, 'B': 0, 'C': 0}, 'C': {'A': 0, 'B': 1, 'C': 0}}
    assert num_walks == expected