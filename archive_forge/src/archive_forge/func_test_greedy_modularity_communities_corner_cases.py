import pytest
import networkx as nx
from networkx.algorithms.community import (
def test_greedy_modularity_communities_corner_cases():
    G = nx.empty_graph()
    assert nx.community.greedy_modularity_communities(G) == []
    G.add_nodes_from(range(3))
    assert nx.community.greedy_modularity_communities(G) == [{0}, {1}, {2}]