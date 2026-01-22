from sympy.core import Add, Mul, Pow
from sympy.core.numbers import (NaN, Infinity, NegativeInfinity, Float, I, pi,
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.elementary.exponential import exp_polar, exp, log
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.functions.special.error_functions import (Ci, Shi, Si, erf, erfc, erfi)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.integrals import meijerint
from sympy.matrices import Matrix
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement
from sympy.polys.domains import QQ, RR
from sympy.polys.polyclasses import DMF
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly
from sympy.polys.matrices import DomainMatrix
from sympy.printing import sstr
from sympy.series.limits import limit
from sympy.series.order import Order
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import nsimplify
from sympy.solvers.solvers import solve
from .recurrence import HolonomicSequence, RecurrenceOperator, RecurrenceOperators
from .holonomicerrors import (NotPowerSeriesError, NotHyperSeriesError,
from sympy.integrals.meijerint import _mytype
def to_hyper(self, as_list=False, _recur=None):
    """
        Returns a hypergeometric function (or linear combination of them)
        representing the given holonomic function.

        Explanation
        ===========

        Returns an answer of the form:
        `a_1 \\cdot x^{b_1} \\cdot{hyper()} + a_2 \\cdot x^{b_2} \\cdot{hyper()} \\dots`

        This is very useful as one can now use ``hyperexpand`` to find the
        symbolic expressions/functions.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import ZZ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')
        >>> # sin(x)
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).to_hyper()
        x*hyper((), (3/2,), -x**2/4)
        >>> # exp(x)
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).to_hyper()
        hyper((), (), x)

        See Also
        ========

        from_hyper, from_meijerg
        """
    if _recur is None:
        recurrence = self.to_sequence()
    else:
        recurrence = _recur
    if isinstance(recurrence, tuple) and len(recurrence) == 2:
        smallest_n = recurrence[1]
        recurrence = recurrence[0]
        constantpower = 0
    elif isinstance(recurrence, tuple) and len(recurrence) == 3:
        smallest_n = recurrence[2]
        constantpower = recurrence[1]
        recurrence = recurrence[0]
    elif len(recurrence) == 1 and len(recurrence[0]) == 2:
        smallest_n = recurrence[0][1]
        recurrence = recurrence[0][0]
        constantpower = 0
    elif len(recurrence) == 1 and len(recurrence[0]) == 3:
        smallest_n = recurrence[0][2]
        constantpower = recurrence[0][1]
        recurrence = recurrence[0][0]
    else:
        sol = self.to_hyper(as_list=as_list, _recur=recurrence[0])
        for i in recurrence[1:]:
            sol += self.to_hyper(as_list=as_list, _recur=i)
        return sol
    u0 = recurrence.u0
    r = recurrence.recurrence
    x = self.x
    x0 = self.x0
    m = r.order
    if m == 0:
        nonzeroterms = roots(r.parent.base.to_sympy(r.listofpoly[0]), recurrence.n, filter='R')
        sol = S.Zero
        for j, i in enumerate(nonzeroterms):
            if i < 0 or int(i) != i:
                continue
            i = int(i)
            if i < len(u0):
                if isinstance(u0[i], (PolyElement, FracElement)):
                    u0[i] = u0[i].as_expr()
                sol += u0[i] * x ** i
            else:
                sol += Symbol('C_%s' % j) * x ** i
        if isinstance(sol, (PolyElement, FracElement)):
            sol = sol.as_expr() * x ** constantpower
        else:
            sol = sol * x ** constantpower
        if as_list:
            if x0 != 0:
                return [(sol.subs(x, x - x0),)]
            return [(sol,)]
        if x0 != 0:
            return sol.subs(x, x - x0)
        return sol
    if smallest_n + m > len(u0):
        raise NotImplementedError("Can't compute sufficient Initial Conditions")
    is_hyper = True
    for i in range(1, len(r.listofpoly) - 1):
        if r.listofpoly[i] != r.parent.base.zero:
            is_hyper = False
            break
    if not is_hyper:
        raise NotHyperSeriesError(self, self.x0)
    a = r.listofpoly[0]
    b = r.listofpoly[-1]
    if isinstance(a.rep[0], (PolyElement, FracElement)):
        c = -(S(a.rep[0].as_expr()) * m ** a.degree()) / (S(b.rep[0].as_expr()) * m ** b.degree())
    else:
        c = -(S(a.rep[0]) * m ** a.degree()) / (S(b.rep[0]) * m ** b.degree())
    sol = 0
    arg1 = roots(r.parent.base.to_sympy(a), recurrence.n)
    arg2 = roots(r.parent.base.to_sympy(b), recurrence.n)
    if as_list:
        listofsol = []
    for i in range(smallest_n + m):
        if i < smallest_n:
            if as_list:
                listofsol.append(((S(u0[i]) * x ** (i + constantpower)).subs(x, x - x0),))
            else:
                sol += S(u0[i]) * x ** i
            continue
        if S(u0[i]) == 0:
            continue
        ap = []
        bq = []
        for k in ordered(arg1.keys()):
            ap.extend([nsimplify((i - k) / m)] * arg1[k])
        for k in ordered(arg2.keys()):
            bq.extend([nsimplify((i - k) / m)] * arg2[k])
        if 1 in bq:
            bq.remove(1)
        else:
            ap.append(1)
        if as_list:
            listofsol.append(((S(u0[i]) * x ** (i + constantpower)).subs(x, x - x0), hyper(ap, bq, c * x ** m).subs(x, x - x0)))
        else:
            sol += S(u0[i]) * hyper(ap, bq, c * x ** m) * x ** i
    if as_list:
        return listofsol
    sol = sol * x ** constantpower
    if x0 != 0:
        return sol.subs(x, x - x0)
    return sol