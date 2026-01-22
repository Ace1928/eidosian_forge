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
def gateinputcount(expr):
    """
    Return the total number of inputs for the logic gates realizing the
    Boolean expression.

    Returns
    =======

    int
        Number of gate inputs

    Note
    ====

    Not all Boolean functions count as gate here, only those that are
    considered to be standard gates. These are: :py:class:`~.And`,
    :py:class:`~.Or`, :py:class:`~.Xor`, :py:class:`~.Not`, and
    :py:class:`~.ITE` (multiplexer). :py:class:`~.Nand`, :py:class:`~.Nor`,
    and :py:class:`~.Xnor` will be evaluated to ``Not(And())`` etc.

    Examples
    ========

    >>> from sympy.logic import And, Or, Nand, Not, gateinputcount
    >>> from sympy.abc import x, y, z
    >>> expr = And(x, y)
    >>> gateinputcount(expr)
    2
    >>> gateinputcount(Or(expr, z))
    4

    Note that ``Nand`` is automatically evaluated to ``Not(And())`` so

    >>> gateinputcount(Nand(x, y, z))
    4
    >>> gateinputcount(Not(And(x, y, z)))
    4

    Although this can be avoided by using ``evaluate=False``

    >>> gateinputcount(Nand(x, y, z, evaluate=False))
    3

    Also note that a comparison will count as a Boolean variable:

    >>> gateinputcount(And(x > z, y >= 2))
    2

    As will a symbol:
    >>> gateinputcount(x)
    0

    """
    if not isinstance(expr, Boolean):
        raise TypeError('Expression must be Boolean')
    if isinstance(expr, BooleanGates):
        return len(expr.args) + sum((gateinputcount(x) for x in expr.args))
    return 0