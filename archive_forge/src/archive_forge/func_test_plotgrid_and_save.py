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
def test_plotgrid_and_save():
    if not matplotlib:
        skip('Matplotlib not the default backend')
    x = Symbol('x')
    y = Symbol('y')
    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        p1 = plot(x)
        p2 = plot_parametric((sin(x), cos(x)), (x, sin(x)), show=False)
        p3 = plot_parametric(cos(x), sin(x), adaptive=False, nb_of_points=500, show=False)
        p4 = plot3d_parametric_line(sin(x), cos(x), x, show=False)
        p = PlotGrid(2, 2, p1, p2, p3, p4)
        filename = 'test_grid1.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = PlotGrid(3, 4, p1, p2, p3, p4)
        filename = 'test_grid2.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p5 = plot(cos(x), (x, -pi, pi), show=False)
        p5[0].line_color = lambda a: a
        p6 = plot(Piecewise((1, x > 0), (0, True)), (x, -1, 1), show=False)
        p7 = plot_contour((x ** 2 + y ** 2, (x, -5, 5), (y, -5, 5)), (x ** 3 + y ** 3, (x, -3, 3), (y, -3, 3)), show=False)
        p = PlotGrid(1, 3, p5, p6, p7)
        filename = 'test_grid3.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()