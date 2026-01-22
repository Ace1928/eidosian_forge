from sympy.core.function import (Function, Lambda, diff, expand_log)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, tan)
from sympy.polys.polytools import (Poly, cancel, factor)
from sympy.integrals.risch import (gcdex_diophantine, frac_in, as_poly_1t,
from sympy.testing.pytest import raises
from sympy.abc import x, t, nu, z, a, y
def test_DifferentialExtension_printing():
    DE = DifferentialExtension(exp(2 * x ** 2) + log(exp(x ** 2) + 1), x)
    assert repr(DE) == "DifferentialExtension(dict([('f', exp(2*x**2) + log(exp(x**2) + 1)), ('x', x), ('T', [x, t0, t1]), ('D', [Poly(1, x, domain='ZZ'), Poly(2*x*t0, t0, domain='ZZ[x]'), Poly(2*t0*x/(t0 + 1), t1, domain='ZZ(x,t0)')]), ('fa', Poly(t1 + t0**2, t1, domain='ZZ[t0]')), ('fd', Poly(1, t1, domain='ZZ')), ('Tfuncs', [Lambda(i, exp(i**2)), Lambda(i, log(t0 + 1))]), ('backsubs', []), ('exts', [None, 'exp', 'log']), ('extargs', [None, x**2, t0 + 1]), ('cases', ['base', 'exp', 'primitive']), ('case', 'primitive'), ('t', t1), ('d', Poly(2*t0*x/(t0 + 1), t1, domain='ZZ(x,t0)')), ('newf', t0**2 + t1), ('level', -1), ('dummy', False)]))"
    assert str(DE) == "DifferentialExtension({fa=Poly(t1 + t0**2, t1, domain='ZZ[t0]'), fd=Poly(1, t1, domain='ZZ'), D=[Poly(1, x, domain='ZZ'), Poly(2*x*t0, t0, domain='ZZ[x]'), Poly(2*t0*x/(t0 + 1), t1, domain='ZZ(x,t0)')]})"