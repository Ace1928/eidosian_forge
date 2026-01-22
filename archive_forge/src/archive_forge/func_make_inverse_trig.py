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
def make_inverse_trig(RuleClass, a, sign_a, c, sign_c, h) -> Rule:
    u_var = Dummy('u')
    rewritten = 1 / sqrt(sign_a * a + sign_c * c * (symbol - h) ** 2)
    quadratic_base = sqrt(c / a) * (symbol - h)
    constant = 1 / sqrt(c)
    u_func = None
    if quadratic_base is not symbol:
        u_func = quadratic_base
        quadratic_base = u_var
    standard_form = 1 / sqrt(sign_a + sign_c * quadratic_base ** 2)
    substep = RuleClass(standard_form, quadratic_base)
    if constant != 1:
        substep = ConstantTimesRule(constant * standard_form, symbol, constant, standard_form, substep)
    if u_func is not None:
        substep = URule(rewritten, symbol, u_var, u_func, substep)
    if h != 0:
        substep = CompleteSquareRule(integrand, symbol, rewritten, substep)
    return substep