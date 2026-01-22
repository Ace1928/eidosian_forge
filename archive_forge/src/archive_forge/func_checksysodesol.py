from sympy.core import S, Pow
from sympy.core.function import (Derivative, AppliedUndef, diff)
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.logic.boolalg import BooleanAtom
from sympy.functions import exp
from sympy.series import Order
from sympy.simplify.simplify import simplify, posify, besselsimp
from sympy.simplify.trigsimp import trigsimp
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.solvers import solve
from sympy.solvers.deutils import _preprocess, ode_order
from sympy.utilities.iterables import iterable, is_sequence
def checksysodesol(eqs, sols, func=None):
    """
    Substitutes corresponding ``sols`` for each functions into each ``eqs`` and
    checks that the result of substitutions for each equation is ``0``. The
    equations and solutions passed can be any iterable.

    This only works when each ``sols`` have one function only, like `x(t)` or `y(t)`.
    For each function, ``sols`` can have a single solution or a list of solutions.
    In most cases it will not be necessary to explicitly identify the function,
    but if the function cannot be inferred from the original equation it
    can be supplied through the ``func`` argument.

    When a sequence of equations is passed, the same sequence is used to return
    the result for each equation with each function substituted with corresponding
    solutions.

    It tries the following method to find zero equivalence for each equation:

    Substitute the solutions for functions, like `x(t)` and `y(t)` into the
    original equations containing those functions.
    This function returns a tuple.  The first item in the tuple is ``True`` if
    the substitution results for each equation is ``0``, and ``False`` otherwise.
    The second item in the tuple is what the substitution results in.  Each element
    of the ``list`` should always be ``0`` corresponding to each equation if the
    first item is ``True``. Note that sometimes this function may return ``False``,
    but with an expression that is identically equal to ``0``, instead of returning
    ``True``.  This is because :py:meth:`~sympy.simplify.simplify.simplify` cannot
    reduce the expression to ``0``.  If an expression returned by each function
    vanishes identically, then ``sols`` really is a solution to ``eqs``.

    If this function seems to hang, it is probably because of a difficult simplification.

    Examples
    ========

    >>> from sympy import Eq, diff, symbols, sin, cos, exp, sqrt, S, Function
    >>> from sympy.solvers.ode.subscheck import checksysodesol
    >>> C1, C2 = symbols('C1:3')
    >>> t = symbols('t')
    >>> x, y = symbols('x, y', cls=Function)
    >>> eq = (Eq(diff(x(t),t), x(t) + y(t) + 17), Eq(diff(y(t),t), -2*x(t) + y(t) + 12))
    >>> sol = [Eq(x(t), (C1*sin(sqrt(2)*t) + C2*cos(sqrt(2)*t))*exp(t) - S(5)/3),
    ... Eq(y(t), (sqrt(2)*C1*cos(sqrt(2)*t) - sqrt(2)*C2*sin(sqrt(2)*t))*exp(t) - S(46)/3)]
    >>> checksysodesol(eq, sol)
    (True, [0, 0])
    >>> eq = (Eq(diff(x(t),t),x(t)*y(t)**4), Eq(diff(y(t),t),y(t)**3))
    >>> sol = [Eq(x(t), C1*exp(-1/(4*(C2 + t)))), Eq(y(t), -sqrt(2)*sqrt(-1/(C2 + t))/2),
    ... Eq(x(t), C1*exp(-1/(4*(C2 + t)))), Eq(y(t), sqrt(2)*sqrt(-1/(C2 + t))/2)]
    >>> checksysodesol(eq, sol)
    (True, [0, 0])

    """

    def _sympify(eq):
        return list(map(sympify, eq if iterable(eq) else [eq]))
    eqs = _sympify(eqs)
    for i in range(len(eqs)):
        if isinstance(eqs[i], Equality):
            eqs[i] = eqs[i].lhs - eqs[i].rhs
    if func is None:
        funcs = []
        for eq in eqs:
            derivs = eq.atoms(Derivative)
            func = set().union(*[d.atoms(AppliedUndef) for d in derivs])
            funcs.extend(func)
        funcs = list(set(funcs))
    if not all((isinstance(func, AppliedUndef) and len(func.args) == 1 for func in funcs)) and len({func.args for func in funcs}) != 1:
        raise ValueError('func must be a function of one variable, not %s' % func)
    for sol in sols:
        if len(sol.atoms(AppliedUndef)) != 1:
            raise ValueError('solutions should have one function only')
    if len(funcs) != len({sol.lhs for sol in sols}):
        raise ValueError('number of solutions provided does not match the number of equations')
    dictsol = {}
    for sol in sols:
        func = list(sol.atoms(AppliedUndef))[0]
        if sol.rhs == func:
            sol = sol.reversed
        solved = sol.lhs == func and (not sol.rhs.has(func))
        if not solved:
            rhs = solve(sol, func)
            if not rhs:
                raise NotImplementedError
        else:
            rhs = sol.rhs
        dictsol[func] = rhs
    checkeq = []
    for eq in eqs:
        for func in funcs:
            eq = sub_func_doit(eq, func, dictsol[func])
        ss = simplify(eq)
        if ss != 0:
            eq = ss.expand(force=True)
            if eq != 0:
                eq = sqrtdenest(eq).simplify()
        else:
            eq = 0
        checkeq.append(eq)
    if len(set(checkeq)) == 1 and list(set(checkeq))[0] == 0:
        return (True, checkeq)
    else:
        return (False, checkeq)