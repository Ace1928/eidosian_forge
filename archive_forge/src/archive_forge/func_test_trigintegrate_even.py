from sympy.core import Ne, Rational, Symbol
from sympy.functions import sin, cos, tan, csc, sec, cot, log, Piecewise
from sympy.integrals.trigonometry import trigintegrate
def test_trigintegrate_even():
    assert trigintegrate(sin(x) ** 2, x) == x / 2 - cos(x) * sin(x) / 2
    assert trigintegrate(cos(x) ** 2, x) == x / 2 + cos(x) * sin(x) / 2
    assert trigintegrate(sin(3 * x) ** 2, x) == x / 2 - cos(3 * x) * sin(3 * x) / 6
    assert trigintegrate(cos(3 * x) ** 2, x) == x / 2 + cos(3 * x) * sin(3 * x) / 6
    assert trigintegrate(sin(x) ** 2 * cos(x) ** 2, x) == x / 8 - sin(2 * x) * cos(2 * x) / 16
    assert trigintegrate(sin(x) ** 4 * cos(x) ** 2, x) == x / 16 - sin(x) * cos(x) / 16 - sin(x) ** 3 * cos(x) / 24 + sin(x) ** 5 * cos(x) / 6
    assert trigintegrate(sin(x) ** 2 * cos(x) ** 4, x) == x / 16 + cos(x) * sin(x) / 16 + cos(x) ** 3 * sin(x) / 24 - cos(x) ** 5 * sin(x) / 6
    assert trigintegrate(sin(x) ** (-4), x) == -2 * cos(x) / (3 * sin(x)) - cos(x) / (3 * sin(x) ** 3)
    assert trigintegrate(cos(x) ** (-6), x) == sin(x) / (5 * cos(x) ** 5) + 4 * sin(x) / (15 * cos(x) ** 3) + 8 * sin(x) / (15 * cos(x))