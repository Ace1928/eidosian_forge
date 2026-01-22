import pytest
import networkx as nx
def test_pseudo_sequence():
    seq = [1000, 3, 3, 3, 3, 2, 2, 2, 1, 1]
    assert nx.is_pseudographical(seq)
    seq = [1000, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    assert not nx.is_pseudographical(seq)
    seq = [1000, 3, 3, 3, 3, 2, 2, -2, 1, 1]
    assert not nx.is_pseudographical(seq)
    seq = [1, 1, 1.1, 1]
    assert not nx.is_pseudographical(seq)
    seq = [1, 1, 'rer', 1]
    assert not nx.is_pseudographical(seq)