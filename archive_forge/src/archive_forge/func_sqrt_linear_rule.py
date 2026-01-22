from __future__ import annotations
from typing import NamedTuple, Type, Callable, Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.containers import Dict
from sympy.core.expr import Expr
from sympy.core.function import Derivative
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Number, E
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne, Boolean
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction, csch,
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (TrigonometricFunction,
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.functions.special.error_functions import (erf, erfi, fresnelc,
from sympy.functions.special.gamma_functions import uppergamma
from sympy.functions.special.elliptic_integrals import elliptic_e, elliptic_f
from sympy.functions.special.polynomials import (chebyshevt, chebyshevu,
from sympy.functions.special.zeta_functions import polylog
from .integrals import Integral
from sympy.logic.boolalg import And
from sympy.ntheory.factor_ import primefactors
from sympy.polys.polytools import degree, lcm_list, gcd_list, Poly
from sympy.simplify.radsimp import fraction
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.strategies.core import switch, do_one, null_safe, condition
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
def sqrt_linear_rule(integral: IntegralInfo):
    """
    Substitute common (a+b*x)**(1/n)
    """
    integrand, x = integral
    a = Wild('a', exclude=[x])
    b = Wild('b', exclude=[x, 0])
    a0 = b0 = 0
    bases, qs, bs = ([], [], [])
    for pow_ in integrand.find(Pow):
        base, exp_ = (pow_.base, pow_.exp)
        if exp_.is_Integer or x not in base.free_symbols:
            continue
        if not exp_.is_Rational:
            return
        match = base.match(a + b * x)
        if not match:
            continue
        a1, b1 = (match[a], match[b])
        if a0 * b1 != a1 * b0 or not (b0 / b1).is_nonnegative:
            return
        if b0 == 0 or (b0 / b1 > 1) is S.true:
            a0, b0 = (a1, b1)
        bases.append(base)
        bs.append(b1)
        qs.append(exp_.q)
    if b0 == 0:
        return
    q0: Integer = lcm_list(qs)
    u_x = (a0 + b0 * x) ** (1 / q0)
    u = Dummy('u')
    substituted = integrand.subs({base ** (S.One / q): (b / b0) ** (S.One / q) * u ** (q0 / q) for base, b, q in zip(bases, bs, qs)}).subs(x, (u ** q0 - a0) / b0)
    substep = integral_steps(substituted * u ** (q0 - 1) * q0 / b0, u)
    if not substep.contains_dont_know():
        step: Rule = URule(integrand, x, u, u_x, substep)
        generic_cond = Ne(b0, 0)
        if generic_cond is not S.true:
            simplified = integrand.subs({b: 0 for b in bs})
            degenerate_step = integral_steps(simplified, x)
            step = PiecewiseRule(integrand, x, [(step, generic_cond), (degenerate_step, S.true)])
        return step