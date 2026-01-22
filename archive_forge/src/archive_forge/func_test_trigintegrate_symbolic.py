from sympy.core import Ne, Rational, Symbol
from sympy.functions import sin, cos, tan, csc, sec, cot, log, Piecewise
from sympy.integrals.trigonometry import trigintegrate
def test_trigintegrate_symbolic():
    n = Symbol('n', integer=True)
    assert trigintegrate(cos(x) ** n, x) is None
    assert trigintegrate(sin(x) ** n, x) is None
    assert trigintegrate(cot(x) ** n, x) is None