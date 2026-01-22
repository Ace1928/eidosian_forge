from sympy.strategies.branch.core import (
def test_do_one():

    def bad(expr):
        raise ValueError
    assert list(do_one(inc)(3)) == [4]
    assert list(do_one(inc, bad)(3)) == [4]
    assert list(do_one(inc, posdec)(3)) == [4]