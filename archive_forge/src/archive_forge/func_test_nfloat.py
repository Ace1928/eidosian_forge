from sympy.concrete.summations import Sum
from sympy.core.basic import Basic, _aresame
from sympy.core.cache import clear_cache
from sympy.core.containers import Dict, Tuple
from sympy.core.expr import Expr, unchanged
from sympy.core.function import (Subs, Function, diff, Lambda, expand,
from sympy.core.numbers import E, Float, zoo, Rational, pi, I, oo, nan
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Dummy, Symbol
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin, cos, acos
from sympy.functions.special.error_functions import expint
from sympy.functions.special.gamma_functions import loggamma, polygamma
from sympy.matrices.dense import Matrix
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.tensor.indexed import Indexed
from sympy.core.function import (PoleError, _mexpand, arity,
from sympy.core.parameters import _exp_is_pow
from sympy.core.sympify import sympify, SympifyError
from sympy.matrices import MutableMatrix, ImmutableMatrix
from sympy.sets.sets import FiniteSet
from sympy.solvers.solveset import solveset
from sympy.tensor.array import NDimArray
from sympy.utilities.iterables import subsets, variations
from sympy.testing.pytest import XFAIL, raises, warns_deprecated_sympy, _both_exp_pow
from sympy.abc import t, w, x, y, z
def test_nfloat():
    from sympy.core.basic import _aresame
    from sympy.polys.rootoftools import rootof
    x = Symbol('x')
    eq = x ** Rational(4, 3) + 4 * x ** (S.One / 3) / 3
    assert _aresame(nfloat(eq), x ** Rational(4, 3) + 4.0 / 3 * x ** (S.One / 3))
    assert _aresame(nfloat(eq, exponent=True), x ** (4.0 / 3) + 4.0 / 3 * x ** (1.0 / 3))
    eq = x ** Rational(4, 3) + 4 * x ** (x / 3) / 3
    assert _aresame(nfloat(eq), x ** Rational(4, 3) + 4.0 / 3 * x ** (x / 3))
    big = 12345678901234567890
    Float_big = Float(big, 15)
    assert _aresame(nfloat(big), Float_big)
    assert _aresame(nfloat(big * x), Float_big * x)
    assert _aresame(nfloat(x ** big, exponent=True), x ** Float_big)
    assert nfloat(cos(x + sqrt(2))) == cos(x + nfloat(sqrt(2)))
    f = S('x*lamda + lamda**3*(x/2 + 1/2) + lamda**2 + 1/4')
    assert not any((a.free_symbols for a in solveset(f.subs(x, -0.139))))
    assert nfloat(-100000 * sqrt(2500000001) + 5000000001) == 9.999999998e-11
    eq = cos(3 * x ** 4 + y) * rootof(x ** 5 + 3 * x ** 3 + 1, 0)
    assert str(nfloat(eq, exponent=False, n=1)) == '-0.7*cos(3.0*x**4 + y)'
    for ti in (dict, Dict):
        d = ti({S.Half: S.Half})
        n = nfloat(d)
        assert isinstance(n, ti)
        assert _aresame(list(n.items()).pop(), (S.Half, Float(0.5)))
    for ti in (dict, Dict):
        d = ti({S.Half: S.Half})
        n = nfloat(d, dkeys=True)
        assert isinstance(n, ti)
        assert _aresame(list(n.items()).pop(), (Float(0.5), Float(0.5)))
    d = [S.Half]
    n = nfloat(d)
    assert type(n) is list
    assert _aresame(n[0], Float(0.5))
    assert _aresame(nfloat(Eq(x, S.Half)).rhs, Float(0.5))
    assert _aresame(nfloat(S(True)), S(True))
    assert _aresame(nfloat(Tuple(S.Half))[0], Float(0.5))
    assert nfloat(Eq((3 - I) ** 2 / 2 + I, 0)) == S.false
    assert nfloat([{S.Half: x}], dkeys=True) == [{Float(0.5): x}]
    A = MutableMatrix([[1, 2], [3, 4]])
    B = MutableMatrix([[Float('1.0', precision=53), Float('2.0', precision=53)], [Float('3.0', precision=53), Float('4.0', precision=53)]])
    assert _aresame(nfloat(A), B)
    A = ImmutableMatrix([[1, 2], [3, 4]])
    B = ImmutableMatrix([[Float('1.0', precision=53), Float('2.0', precision=53)], [Float('3.0', precision=53), Float('4.0', precision=53)]])
    assert _aresame(nfloat(A), B)
    f = Function('f')
    assert not nfloat(f(2)).atoms(Float)