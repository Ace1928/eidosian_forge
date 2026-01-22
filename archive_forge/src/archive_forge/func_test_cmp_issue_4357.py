from sympy.core.numbers import Rational
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.exponential import exp
def test_cmp_issue_4357():
    """ Check that Basic subclasses can be compared with sympifiable objects.

    https://github.com/sympy/sympy/issues/4357
    """
    assert not Symbol == 1
    assert Symbol != 1
    assert not Symbol == 'x'
    assert Symbol != 'x'