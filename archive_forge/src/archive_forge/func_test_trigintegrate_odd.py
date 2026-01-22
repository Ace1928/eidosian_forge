from sympy.core import Ne, Rational, Symbol
from sympy.functions import sin, cos, tan, csc, sec, cot, log, Piecewise
from sympy.integrals.trigonometry import trigintegrate
def test_trigintegrate_odd():
    assert trigintegrate(Rational(1), x) == x
    assert trigintegrate(x, x) is None
    assert trigintegrate(x ** 2, x) is None
    assert trigintegrate(sin(x), x) == -cos(x)
    assert trigintegrate(cos(x), x) == sin(x)
    assert trigintegrate(sin(3 * x), x) == -cos(3 * x) / 3
    assert trigintegrate(cos(3 * x), x) == sin(3 * x) / 3
    y = Symbol('y')
    assert trigintegrate(sin(y * x), x) == Piecewise((-cos(y * x) / y, Ne(y, 0)), (0, True))
    assert trigintegrate(cos(y * x), x) == Piecewise((sin(y * x) / y, Ne(y, 0)), (x, True))
    assert trigintegrate(sin(y * x) ** 2, x) == Piecewise(((x * y / 2 - sin(x * y) * cos(x * y) / 2) / y, Ne(y, 0)), (0, True))
    assert trigintegrate(sin(y * x) * cos(y * x), x) == Piecewise((sin(x * y) ** 2 / (2 * y), Ne(y, 0)), (0, True))
    assert trigintegrate(cos(y * x) ** 2, x) == Piecewise(((x * y / 2 + sin(x * y) * cos(x * y) / 2) / y, Ne(y, 0)), (x, True))
    y = Symbol('y', positive=True)
    assert trigintegrate(sin(y * x), x, conds='none') == -cos(y * x) / y
    assert trigintegrate(cos(y * x), x, conds='none') == sin(y * x) / y
    assert trigintegrate(sin(x) * cos(x), x) == sin(x) ** 2 / 2
    assert trigintegrate(sin(x) * cos(x) ** 2, x) == -cos(x) ** 3 / 3
    assert trigintegrate(sin(x) ** 2 * cos(x), x) == sin(x) ** 3 / 3
    assert trigintegrate(sin(x) ** 7 * cos(x), x) == sin(x) ** 8 / 8
    assert trigintegrate(sin(x) * cos(x) ** 7, x) == -cos(x) ** 8 / 8
    assert trigintegrate(sin(x) ** 7 * cos(x) ** 3, x) == -sin(x) ** 10 / 10 + sin(x) ** 8 / 8
    assert trigintegrate(sin(x) ** 3 * cos(x) ** 7, x) == cos(x) ** 10 / 10 - cos(x) ** 8 / 8
    assert trigintegrate(sin(x) ** (-1) * cos(x) ** (-1), x) == -log(sin(x) ** 2 - 1) / 2 + log(sin(x))