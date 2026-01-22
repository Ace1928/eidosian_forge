from itertools import zip_longest
from sympy.utilities.enumerative import (
from sympy.utilities.iterables import _set_partitions
def test_multiset_partitions_taocp():
    """Compares the output of multiset_partitions_taocp with a baseline
    (set partition based) implementation."""
    multiplicities = [2, 2]
    compare_multiset_w_baseline(multiplicities)
    multiplicities = [4, 3, 1]
    compare_multiset_w_baseline(multiplicities)