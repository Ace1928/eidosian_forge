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
def quadratic_denom_rule(integral):
    integrand, symbol = integral
    a = Wild('a', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    c = Wild('c', exclude=[symbol])
    match = integrand.match(a / (b * symbol ** 2 + c))
    if match:
        a, b, c = (match[a], match[b], match[c])
        general_rule = ArctanRule(integrand, symbol, a, b, c)
        if b.is_extended_real and c.is_extended_real:
            positive_cond = c / b > 0
            if positive_cond is S.true:
                return general_rule
            coeff = a / (2 * sqrt(-c) * sqrt(b))
            constant = sqrt(-c / b)
            r1 = 1 / (symbol - constant)
            r2 = 1 / (symbol + constant)
            log_steps = [ReciprocalRule(r1, symbol, symbol - constant), ConstantTimesRule(-r2, symbol, -1, r2, ReciprocalRule(r2, symbol, symbol + constant))]
            rewritten = sub = r1 - r2
            negative_step = AddRule(sub, symbol, log_steps)
            if coeff != 1:
                rewritten = Mul(coeff, sub, evaluate=False)
                negative_step = ConstantTimesRule(rewritten, symbol, coeff, sub, negative_step)
            negative_step = RewriteRule(integrand, symbol, rewritten, negative_step)
            if positive_cond is S.false:
                return negative_step
            return PiecewiseRule(integrand, symbol, [(general_rule, positive_cond), (negative_step, S.true)])
        return general_rule
    d = Wild('d', exclude=[symbol])
    match2 = integrand.match(a / (b * symbol ** 2 + c * symbol + d))
    if match2:
        b, c = (match2[b], match2[c])
        if b.is_zero:
            return
        u = Dummy('u')
        u_func = symbol + c / (2 * b)
        integrand2 = integrand.subs(symbol, u - c / (2 * b))
        next_step = integral_steps(integrand2, u)
        if next_step:
            return URule(integrand2, symbol, u, u_func, next_step)
        else:
            return
    e = Wild('e', exclude=[symbol])
    match3 = integrand.match((a * symbol + b) / (c * symbol ** 2 + d * symbol + e))
    if match3:
        a, b, c, d, e = (match3[a], match3[b], match3[c], match3[d], match3[e])
        if c.is_zero:
            return
        denominator = c * symbol ** 2 + d * symbol + e
        const = a / (2 * c)
        numer1 = 2 * c * symbol + d
        numer2 = -const * d + b
        u = Dummy('u')
        step1 = URule(integrand, symbol, u, denominator, integral_steps(u ** (-1), u))
        if const != 1:
            step1 = ConstantTimesRule(const * numer1 / denominator, symbol, const, numer1 / denominator, step1)
        if numer2.is_zero:
            return step1
        step2 = integral_steps(numer2 / denominator, symbol)
        substeps = AddRule(integrand, symbol, [step1, step2])
        rewriten = const * numer1 / denominator + numer2 / denominator
        return RewriteRule(integrand, symbol, rewriten, substeps)
    return