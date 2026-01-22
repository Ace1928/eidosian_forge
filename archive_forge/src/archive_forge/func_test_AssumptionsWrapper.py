from sympy.assumptions.ask import Q
from sympy.assumptions.wrapper import (AssumptionsWrapper, is_infinite,
from sympy.core.symbol import Symbol
from sympy.core.assumptions import _assume_defined
def test_AssumptionsWrapper():
    x = Symbol('x', positive=True)
    y = Symbol('y')
    assert AssumptionsWrapper(x).is_positive
    assert AssumptionsWrapper(y).is_positive is None
    assert AssumptionsWrapper(y, Q.positive(y)).is_positive