from sympy.strategies.branch.core import (
def test_yieldify():
    yinc = yieldify(lambda x: x + 1)
    assert list(yinc(3)) == [4]