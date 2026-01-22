from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent
def bool_map(bool1, bool2):
    """
    Return the simplified version of *bool1*, and the mapping of variables
    that makes the two expressions *bool1* and *bool2* represent the same
    logical behaviour for some correspondence between the variables
    of each.
    If more than one mappings of this sort exist, one of them
    is returned.

    For example, ``And(x, y)`` is logically equivalent to ``And(a, b)`` for
    the mapping ``{x: a, y: b}`` or ``{x: b, y: a}``.
    If no such mapping exists, return ``False``.

    Examples
    ========

    >>> from sympy import SOPform, bool_map, Or, And, Not, Xor
    >>> from sympy.abc import w, x, y, z, a, b, c, d
    >>> function1 = SOPform([x, z, y],[[1, 0, 1], [0, 0, 1]])
    >>> function2 = SOPform([a, b, c],[[1, 0, 1], [1, 0, 0]])
    >>> bool_map(function1, function2)
    (y & ~z, {y: a, z: b})

    The results are not necessarily unique, but they are canonical. Here,
    ``(w, z)`` could be ``(a, d)`` or ``(d, a)``:

    >>> eq =  Or(And(Not(y), w), And(Not(y), z), And(x, y))
    >>> eq2 = Or(And(Not(c), a), And(Not(c), d), And(b, c))
    >>> bool_map(eq, eq2)
    ((x & y) | (w & ~y) | (z & ~y), {w: a, x: b, y: c, z: d})
    >>> eq = And(Xor(a, b), c, And(c,d))
    >>> bool_map(eq, eq.subs(c, x))
    (c & d & (a | b) & (~a | ~b), {a: a, b: b, c: d, d: x})

    """

    def match(function1, function2):
        """Return the mapping that equates variables between two
        simplified boolean expressions if possible.

        By "simplified" we mean that a function has been denested
        and is either an And (or an Or) whose arguments are either
        symbols (x), negated symbols (Not(x)), or Or (or an And) whose
        arguments are only symbols or negated symbols. For example,
        ``And(x, Not(y), Or(w, Not(z)))``.

        Basic.match is not robust enough (see issue 4835) so this is
        a workaround that is valid for simplified boolean expressions
        """
        if function1.__class__ != function2.__class__:
            return None
        if len(function1.args) != len(function2.args):
            return None
        if function1.is_Symbol:
            return {function1: function2}
        f1 = _finger(function1)
        f2 = _finger(function2)
        if len(f1) != len(f2):
            return False
        matchdict = {}
        for k in f1.keys():
            if k not in f2:
                return False
            if len(f1[k]) != len(f2[k]):
                return False
            for i, x in enumerate(f1[k]):
                matchdict[x] = f2[k][i]
        return matchdict
    a = simplify_logic(bool1)
    b = simplify_logic(bool2)
    m = match(a, b)
    if m:
        return (a, m)
    return m