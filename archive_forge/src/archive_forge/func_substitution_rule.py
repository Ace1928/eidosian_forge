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
def substitution_rule(integral):
    integrand, symbol = integral
    u_var = Dummy('u')
    substitutions = find_substitutions(integrand, symbol, u_var)
    count = 0
    if substitutions:
        debug('List of Substitution Rules')
        ways = []
        for u_func, c, substituted in substitutions:
            subrule = integral_steps(substituted, u_var)
            count = count + 1
            debug('Rule {}: {}'.format(count, subrule))
            if subrule.contains_dont_know():
                continue
            if simplify(c - 1) != 0:
                _, denom = c.as_numer_denom()
                if subrule:
                    subrule = ConstantTimesRule(c * substituted, u_var, c, substituted, subrule)
                if denom.free_symbols:
                    piecewise = []
                    could_be_zero = []
                    if isinstance(denom, Mul):
                        could_be_zero = denom.args
                    else:
                        could_be_zero.append(denom)
                    for expr in could_be_zero:
                        if not fuzzy_not(expr.is_zero):
                            substep = integral_steps(manual_subs(integrand, expr, 0), symbol)
                            if substep:
                                piecewise.append((substep, Eq(expr, 0)))
                    piecewise.append((subrule, True))
                    subrule = PiecewiseRule(substituted, symbol, piecewise)
            ways.append(URule(integrand, symbol, u_var, u_func, subrule))
        if len(ways) > 1:
            return AlternativeRule(integrand, symbol, ways)
        elif ways:
            return ways[0]