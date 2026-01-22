from sympy.strategies.branch.core import (
def test_sfilter():
    brl = sfilter(even, one_to_n)
    assert set(brl(10)) == {0, 2, 4, 6, 8}