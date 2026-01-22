import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
def test_all_pairs_connectivity_icosahedral(self):
    G = nx.icosahedral_graph()
    C = nx.all_pairs_node_connectivity(G)
    assert all((5 == C[u][v] for u, v in itertools.combinations(G, 2)))