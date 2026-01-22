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
def test_plot_and_save_4():
    if not matplotlib:
        skip('Matplotlib not the default backend')
    x = Symbol('x')
    y = Symbol('y')
    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        with warns(UserWarning, match='The evaluation of the expression is problematic', test_stacklevel=False):
            i = Integral(log((sin(x) ** 2 + 1) * sqrt(x ** 2 + 1)), (x, 0, y))
            p = plot(i, (y, 1, 5))
            filename = 'test_advanced_integral.png'
            p.save(os.path.join(tmpdir, filename))
            p._backend.close()