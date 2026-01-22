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
@classmethod
def unrank_lex(cls, size, rank):
    """
        Lexicographic permutation unranking.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> a = Permutation.unrank_lex(5, 10)
        >>> a.rank()
        10
        >>> a
        Permutation([0, 2, 4, 1, 3])

        See Also
        ========

        rank, next_lex
        """
    perm_array = [0] * size
    psize = 1
    for i in range(size):
        new_psize = psize * (i + 1)
        d = rank % new_psize // psize
        rank -= d * psize
        perm_array[size - i - 1] = d
        for j in range(size - i, size):
            if perm_array[j] > d - 1:
                perm_array[j] += 1
        psize = new_psize
    return cls._af_new(perm_array)