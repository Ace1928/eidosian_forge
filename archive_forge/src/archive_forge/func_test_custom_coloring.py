import os
from tempfile import TemporaryDirectory
from sympy.concrete.summations import Sum
from sympy.core.numbers import (I, oo, pi)
from sympy.core.relational import Ne
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import (real_root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.hyper import meijerg
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.plotting.plot import (
from sympy.plotting.plot import (
from sympy.testing.pytest import skip, raises, warns, warns_deprecated_sympy
from sympy.utilities import lambdify as lambdify_
from sympy.utilities.exceptions import ignore_warnings
def test_custom_coloring():
    x = Symbol('x')
    y = Symbol('y')
    plot(cos(x), line_color=lambda a: a)
    plot(cos(x), line_color=1)
    plot(cos(x), line_color='r')
    plot_parametric(cos(x), sin(x), line_color=lambda a: a)
    plot_parametric(cos(x), sin(x), line_color=1)
    plot_parametric(cos(x), sin(x), line_color='r')
    plot3d_parametric_line(cos(x), sin(x), x, line_color=lambda a: a)
    plot3d_parametric_line(cos(x), sin(x), x, line_color=1)
    plot3d_parametric_line(cos(x), sin(x), x, line_color='r')
    plot3d_parametric_surface(cos(x + y), sin(x - y), x - y, (x, -5, 5), (y, -5, 5), surface_color=lambda a, b: a ** 2 + b ** 2)
    plot3d_parametric_surface(cos(x + y), sin(x - y), x - y, (x, -5, 5), (y, -5, 5), surface_color=1)
    plot3d_parametric_surface(cos(x + y), sin(x - y), x - y, (x, -5, 5), (y, -5, 5), surface_color='r')
    plot3d(x * y, (x, -5, 5), (y, -5, 5), surface_color=lambda a, b: a ** 2 + b ** 2)
    plot3d(x * y, (x, -5, 5), (y, -5, 5), surface_color=1)
    plot3d(x * y, (x, -5, 5), (y, -5, 5), surface_color='r')