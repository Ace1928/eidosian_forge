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
def test_sawtooth_wave():
    s = fourier_series(x, (x, 0, pi))
    assert s.truncate(4) == pi / 2 - sin(2 * x) - sin(4 * x) / 2 - sin(6 * x) / 3
    s = fourier_series(x, (x, 0, 1))
    assert s.truncate(4) == S.Half - sin(2 * pi * x) / pi - sin(4 * pi * x) / (2 * pi) - sin(6 * pi * x) / (3 * pi)