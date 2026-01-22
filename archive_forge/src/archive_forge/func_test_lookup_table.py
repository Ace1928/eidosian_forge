from sympy.core.function import expand_func
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.complexes import Abs, arg, re, unpolarify
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import cosh, acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (cos, sin, sinc, asin)
from sympy.functions.special.error_functions import (erf, erfc)
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import simplify
from sympy.integrals.meijerint import (_rewrite_single, _rewrite1,
from sympy.testing.pytest import slow
from sympy.core.random import (verify_numerically,
from sympy.abc import x, y, a, b, c, d, s, t, z
@slow
def test_lookup_table():
    from sympy.core.random import uniform, randrange
    from sympy.core.add import Add
    from sympy.integrals.meijerint import z as z_dummy
    table = {}
    _create_lookup_table(table)
    for _, l in table.items():
        for formula, terms, cond, hint in sorted(l, key=default_sort_key):
            subs = {}
            for ai in list(formula.free_symbols) + [z_dummy]:
                if hasattr(ai, 'properties') and ai.properties:
                    subs[ai] = randrange(1, 10)
                else:
                    subs[ai] = uniform(1.5, 2.0)
            if not isinstance(terms, list):
                terms = terms(subs)
            expanded = [hyperexpand(g) for _, g in terms]
            assert all((x.is_Piecewise or not x.has(meijerg) for x in expanded))
            expanded = Add(*[f * x for f, x in terms])
            a, b = (formula.n(subs=subs), expanded.n(subs=subs))
            r = min(abs(a), abs(b))
            if r < 1:
                assert abs(a - b).n() <= 1e-10
            else:
                assert (abs(a - b) / r).n() <= 1e-10