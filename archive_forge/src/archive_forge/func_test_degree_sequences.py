import pytest
from networkx.utils import (
def test_degree_sequences():
    seq = powerlaw_sequence(10, seed=1)
    seq = powerlaw_sequence(10)
    assert len(seq) == 10