from sympy.strategies.branch.core import (
def test_notempty():

    def ident_if_even(x):
        if even(x):
            yield x
    brl = notempty(ident_if_even)
    assert set(brl(4)) == {4}
    assert set(brl(5)) == {5}