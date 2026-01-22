import random
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from sympy.core.parameters import global_parameters
from sympy.core.basic import Atom
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.sympify import _sympify
from sympy.matrices import zeros
from sympy.polys.polytools import lcm
from sympy.printing.repr import srepr
from sympy.utilities.iterables import (flatten, has_variety, minlex,
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import ifac
from sympy.multipledispatch import dispatch
def get_precedence_distance(self, other):
    """
        Computes the precedence distance between two permutations.

        Explanation
        ===========

        Suppose p and p' represent n jobs. The precedence metric
        counts the number of times a job j is preceded by job i
        in both p and p'. This metric is commutative.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([2, 0, 4, 3, 1])
        >>> q = Permutation([3, 1, 2, 4, 0])
        >>> p.get_precedence_distance(q)
        7
        >>> q.get_precedence_distance(p)
        7

        See Also
        ========

        get_precedence_matrix, get_adjacency_matrix, get_adjacency_distance
        """
    if self.size != other.size:
        raise ValueError('The permutations must be of equal size.')
    self_prec_mat = self.get_precedence_matrix()
    other_prec_mat = other.get_precedence_matrix()
    n_prec = 0
    for i in range(self.size):
        for j in range(self.size):
            if i == j:
                continue
            if self_prec_mat[i, j] * other_prec_mat[i, j] == 1:
                n_prec += 1
    d = self.size * (self.size - 1) // 2 - n_prec
    return d