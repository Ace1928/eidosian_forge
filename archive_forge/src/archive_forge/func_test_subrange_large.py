from itertools import zip_longest
from sympy.utilities.enumerative import (
from sympy.utilities.iterables import _set_partitions
def test_subrange_large():
    mult = [6, 3, 2, 1]
    lb = 4
    ub = 7
    subrange_exercise(mult, lb, ub)