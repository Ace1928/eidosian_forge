from sympy.strategies.branch.core import (
def test_multiplex():
    brl = multiplex(posdec, branch5)
    assert set(brl(3)) == {2}
    assert set(brl(7)) == {6, 8}
    assert set(brl(5)) == {4, 6}