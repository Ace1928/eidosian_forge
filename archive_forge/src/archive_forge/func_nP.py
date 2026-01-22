from math import prod
from collections import defaultdict
from typing import Tuple as tTuple
from sympy.core import S, Symbol, Add, Dummy
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import ArgumentIndexError, Function, expand_mul
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import E, I, pi, oo, Rational, Integer
from sympy.core.relational import Eq, is_le, is_gt
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions.combinatorial.factorials import (binomial,
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.ntheory.primetest import isprime, is_square
from sympy.polys.appellseqs import bernoulli_poly, euler_poly, genocchi_poly
from sympy.utilities.enumerative import MultisetPartitionTraverser
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import multiset, multiset_derangements, iterable
from sympy.utilities.memoization import recurrence_memo
from sympy.utilities.misc import as_int
from mpmath import mp, workprec
from mpmath.libmp import ifib as _ifib
def nP(n, k=None, replacement=False):
    """Return the number of permutations of ``n`` items taken ``k`` at a time.

    Possible values for ``n``:

        integer - set of length ``n``

        sequence - converted to a multiset internally

        multiset - {element: multiplicity}

    If ``k`` is None then the total of all permutations of length 0
    through the number of items represented by ``n`` will be returned.

    If ``replacement`` is True then a given item can appear more than once
    in the ``k`` items. (For example, for 'ab' permutations of 2 would
    include 'aa', 'ab', 'ba' and 'bb'.) The multiplicity of elements in
    ``n`` is ignored when ``replacement`` is True but the total number
    of elements is considered since no element can appear more times than
    the number of elements in ``n``.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import nP
    >>> from sympy.utilities.iterables import multiset_permutations, multiset
    >>> nP(3, 2)
    6
    >>> nP('abc', 2) == nP(multiset('abc'), 2) == 6
    True
    >>> nP('aab', 2)
    3
    >>> nP([1, 2, 2], 2)
    3
    >>> [nP(3, i) for i in range(4)]
    [1, 3, 6, 6]
    >>> nP(3) == sum(_)
    True

    When ``replacement`` is True, each item can have multiplicity
    equal to the length represented by ``n``:

    >>> nP('aabc', replacement=True)
    121
    >>> [len(list(multiset_permutations('aaaabbbbcccc', i))) for i in range(5)]
    [1, 3, 9, 27, 81]
    >>> sum(_)
    121

    See Also
    ========
    sympy.utilities.iterables.multiset_permutations

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Permutation

    """
    try:
        n = as_int(n)
    except ValueError:
        return Integer(_nP(_multiset_histogram(n), k, replacement))
    return Integer(_nP(n, k, replacement))