from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (true, false, Eq, Ne, Ge, Gt, Le, Lt, Symbol,
def test_Contains():
    assert Contains(x, FiniteSet(0)) != false
    assert Contains(x, Interval(1, 1)) != false
    assert Contains(oo, Interval(-oo, oo)) == false
    assert Contains(-oo, Interval(-oo, oo)) == false