from itertools import combinations
import pytest
from networkx import DiGraph
from networkx.algorithms.tournament import (
def test_condition_not_satisfied():
    condition = lambda x: x > 0
    iter_in = [0]
    assert index_satisfying(iter_in, condition) == 1