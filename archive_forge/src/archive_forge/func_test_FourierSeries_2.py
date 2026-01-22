from sympy.core.add import Add
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin, sinc, tan)
from sympy.series.fourier import fourier_series
from sympy.series.fourier import FourierSeries
from sympy.testing.pytest import raises
from functools import lru_cache
def test_FourierSeries_2():
    p = Piecewise((0, x < 0), (x, True))
    f = fourier_series(p, (x, -2, 2))
    assert f.term(3) == 2 * sin(3 * pi * x / 2) / (3 * pi) - 4 * cos(3 * pi * x / 2) / (9 * pi ** 2)
    assert f.truncate() == 2 * sin(pi * x / 2) / pi - sin(pi * x) / pi - 4 * cos(pi * x / 2) / pi ** 2 + S.Half