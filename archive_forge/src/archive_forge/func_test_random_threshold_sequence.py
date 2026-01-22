import pytest
import networkx as nx
import networkx.algorithms.threshold as nxt
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test_random_threshold_sequence(self):
    assert len(nxt.random_threshold_sequence(10, 0.5)) == 10
    assert nxt.random_threshold_sequence(10, 0.5, seed=42) == ['d', 'i', 'd', 'd', 'd', 'i', 'i', 'i', 'd', 'd']
    assert pytest.raises(ValueError, nxt.random_threshold_sequence, 10, 1.5)