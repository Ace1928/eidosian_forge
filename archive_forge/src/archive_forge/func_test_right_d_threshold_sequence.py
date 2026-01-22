import pytest
import networkx as nx
import networkx.algorithms.threshold as nxt
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test_right_d_threshold_sequence(self):
    assert nxt.right_d_threshold_sequence(3, 2) == ['d', 'i', 'd']
    assert pytest.raises(ValueError, nxt.right_d_threshold_sequence, 2, 3)