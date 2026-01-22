from sympy.core import S, Function, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.core.numbers import Rational, NumberSymbol, _illegal
from sympy.core.parameters import global_parameters
from sympy.core.relational import (Lt, Gt, Eq, Ne, Relational,
from sympy.core.sorting import ordered
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
from sympy.utilities.iterables import uniq, sift, common_prefix
from sympy.utilities.misc import filldedent, func_name
from itertools import product
def piecewise_simplify_arguments(expr, **kwargs):
    from sympy.simplify.simplify import simplify
    f1 = expr.args[0].cond.free_symbols
    args = None
    if len(f1) == 1 and (not expr.atoms(Eq)):
        x = f1.pop()
        ok, abe_ = expr._intervals(x, err_on_Eq=True)

        def include(c, x, a):
            """return True if c.subs(x, a) is True, else False"""
            try:
                return c.subs(x, a) == True
            except TypeError:
                return False
        if ok:
            args = []
            covered = S.EmptySet
            from sympy.sets.sets import Interval
            for a, b, e, i in abe_:
                c = expr.args[i].cond
                incl_a = include(c, x, a)
                incl_b = include(c, x, b)
                iv = Interval(a, b, not incl_a, not incl_b)
                cset = iv - covered
                if not cset:
                    continue
                if incl_a and incl_b:
                    if a.is_infinite and b.is_infinite:
                        c = S.true
                    elif b.is_infinite:
                        c = x >= a
                    elif a in covered or a.is_infinite:
                        c = x <= b
                    else:
                        c = And(a <= x, x <= b)
                elif incl_a:
                    if a in covered or a.is_infinite:
                        c = x < b
                    else:
                        c = And(a <= x, x < b)
                elif incl_b:
                    if b.is_infinite:
                        c = x > a
                    else:
                        c = x <= b
                elif a in covered:
                    c = x < b
                else:
                    c = And(a < x, x < b)
                covered |= iv
                if a is S.NegativeInfinity and incl_a:
                    covered |= {S.NegativeInfinity}
                if b is S.Infinity and incl_b:
                    covered |= {S.Infinity}
                args.append((e, c))
            if not S.Reals.is_subset(covered):
                args.append((Undefined, True))
    if args is None:
        args = list(expr.args)
        for i in range(len(args)):
            e, c = args[i]
            if isinstance(c, Basic):
                c = simplify(c, **kwargs)
            args[i] = (e, c)
    doit = kwargs.pop('doit', None)
    for i in range(len(args)):
        e, c = args[i]
        if isinstance(e, Basic):
            newe = simplify(e, doit=False, **kwargs)
            if newe != e:
                e = newe
        args[i] = (e, c)
    if doit is not None:
        kwargs['doit'] = doit
    return Piecewise(*args)