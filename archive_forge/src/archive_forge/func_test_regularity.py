import pytest
import networkx as nx
from networkx.classes import Graph, MultiDiGraph
from networkx.generators.directed import (
def test_regularity(self):
    """Tests that the generated graph is `k`-out-regular."""
    n = 10
    k = 3
    G = random_uniform_k_out_graph(n, k)
    assert all((d == k for v, d in G.out_degree()))
    G = random_uniform_k_out_graph(n, k, seed=42)
    assert all((d == k for v, d in G.out_degree()))