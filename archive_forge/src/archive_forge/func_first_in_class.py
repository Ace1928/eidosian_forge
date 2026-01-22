from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.combinatorics.free_groups import (FreeGroup, FreeGroupElement,
from sympy.combinatorics.rewritingsystem import RewritingSystem
from sympy.combinatorics.coset_table import (CosetTable,
from sympy.combinatorics import PermutationGroup
from sympy.matrices.normalforms import invariant_factors
from sympy.matrices import Matrix
from sympy.polys.polytools import gcd
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.magic import pollute
from itertools import product
def first_in_class(C, Y=()):
    """
    Checks whether the subgroup ``H=G1`` corresponding to the Coset Table
    could possibly be the canonical representative of its conjugacy class.

    Parameters
    ==========

    C: CosetTable

    Returns
    =======

    bool: True/False

    If this returns False, then no descendant of C can have that property, and
    so we can abandon C. If it returns True, then we need to process further
    the node of the search tree corresponding to C, and so we call
    ``descendant_subgroups`` recursively on C.

    Examples
    ========

    >>> from sympy.combinatorics import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, CosetTable, first_in_class
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**2, y**3, (x*y)**4])
    >>> C = CosetTable(f, [])
    >>> C.table = [[0, 0, None, None]]
    >>> first_in_class(C)
    True
    >>> C.table = [[1, 1, 1, None], [0, 0, None, 1]]; C.p = [0, 1]
    >>> first_in_class(C)
    True
    >>> C.table = [[1, 1, 2, 1], [0, 0, 0, None], [None, None, None, 0]]
    >>> C.p = [0, 1, 2]
    >>> first_in_class(C)
    False
    >>> C.table = [[1, 1, 1, 2], [0, 0, 2, 0], [2, None, 0, 1]]
    >>> first_in_class(C)
    False

    # TODO:: Sims points out in [Sim94] that performance can be improved by
    # remembering some of the information computed by ``first_in_class``. If
    # the ``continue alpha`` statement is executed at line 14, then the same thing
    # will happen for that value of alpha in any descendant of the table C, and so
    # the values the values of alpha for which this occurs could profitably be
    # stored and passed through to the descendants of C. Of course this would
    # make the code more complicated.

    # The code below is taken directly from the function on page 208 of [Sim94]
    # nu[alpha]

    """
    n = C.n
    lamda = -1
    nu = [None] * n
    mu = [None] * n
    next_alpha = False
    for alpha in range(1, n):
        for beta in range(lamda + 1):
            nu[mu[beta]] = None
        for w in Y:
            if C.table[alpha][C.A_dict[w]] != alpha:
                next_alpha = True
                break
        if next_alpha:
            next_alpha = False
            continue
        mu[0] = alpha
        nu[alpha] = 0
        lamda = 0
        for beta in range(n):
            for x in C.A:
                gamma = C.table[beta][C.A_dict[x]]
                delta = C.table[mu[beta]][C.A_dict[x]]
                if gamma is None or delta is None:
                    next_alpha = True
                    break
                if nu[delta] is None:
                    lamda += 1
                    nu[delta] = lamda
                    mu[lamda] = delta
                if nu[delta] < gamma:
                    return False
                if nu[delta] > gamma:
                    next_alpha = True
                    break
            if next_alpha:
                next_alpha = False
                break
    return True