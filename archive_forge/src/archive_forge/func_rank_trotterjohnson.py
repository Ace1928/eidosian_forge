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
def rank_trotterjohnson(self):
    """
        Returns the Trotter Johnson rank, which we get from the minimal
        change algorithm. See [4] section 2.4.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.rank_trotterjohnson()
        0
        >>> p = Permutation([0, 2, 1, 3])
        >>> p.rank_trotterjohnson()
        7

        See Also
        ========

        unrank_trotterjohnson, next_trotterjohnson
        """
    if self.array_form == [] or self.is_Identity:
        return 0
    if self.array_form == [1, 0]:
        return 1
    perm = self.array_form
    n = self.size
    rank = 0
    for j in range(1, n):
        k = 1
        i = 0
        while perm[i] != j:
            if perm[i] < j:
                k += 1
            i += 1
        j1 = j + 1
        if rank % 2 == 0:
            rank = j1 * rank + j1 - k
        else:
            rank = j1 * rank + k - 1
    return rank