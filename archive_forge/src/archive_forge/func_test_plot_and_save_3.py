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
def test_plot_and_save_3():
    if not matplotlib:
        skip('Matplotlib not the default backend')
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        p = plot(sin(x))
        p[0].line_color = lambda a: a
        filename = 'test_colors_line_arity1.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].line_color = lambda a, b: b
        filename = 'test_colors_line_arity2.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot(x * sin(x), x * cos(x), (x, 0, 10))
        p[0].line_color = lambda a: a
        filename = 'test_colors_param_line_arity1.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].line_color = lambda a, b: a
        filename = 'test_colors_param_line_arity1.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].line_color = lambda a, b: b
        filename = 'test_colors_param_line_arity2b.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot3d_parametric_line(sin(x) + 0.1 * sin(x) * cos(7 * x), cos(x) + 0.1 * cos(x) * cos(7 * x), 0.1 * sin(7 * x), (x, 0, 2 * pi))
        p[0].line_color = lambdify_(x, sin(4 * x))
        filename = 'test_colors_3d_line_arity1.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].line_color = lambda a, b: b
        filename = 'test_colors_3d_line_arity2.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].line_color = lambda a, b, c: c
        filename = 'test_colors_3d_line_arity3.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot3d(sin(x) * y, (x, 0, 6 * pi), (y, -5, 5))
        p[0].surface_color = lambda a: a
        filename = 'test_colors_surface_arity1.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambda a, b: b
        filename = 'test_colors_surface_arity2.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambda a, b, c: c
        filename = 'test_colors_surface_arity3a.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambdify_((x, y, z), sqrt((x - 3 * pi) ** 2 + y ** 2))
        filename = 'test_colors_surface_arity3b.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot3d_parametric_surface(x * cos(4 * y), x * sin(4 * y), y, (x, -1, 1), (y, -1, 1))
        p[0].surface_color = lambda a: a
        filename = 'test_colors_param_surf_arity1.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambda a, b: a * b
        filename = 'test_colors_param_surf_arity2.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambdify_((x, y, z), sqrt(x ** 2 + y ** 2 + z ** 2))
        filename = 'test_colors_param_surf_arity3.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()