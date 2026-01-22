import pytest
from networkx.exception import NetworkXError
from networkx.generators.duplication import (
def test_final_size(self):
    N = 10
    n = 5
    p = 0.5
    q = 0.5
    G = partial_duplication_graph(N, n, p, q)
    assert len(G) == N
    G = partial_duplication_graph(N, n, p, q, seed=42)
    assert len(G) == N