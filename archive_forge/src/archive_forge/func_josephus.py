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
def josephus(cls, m, n, s=1):
    """Return as a permutation the shuffling of range(n) using the Josephus
        scheme in which every m-th item is selected until all have been chosen.
        The returned permutation has elements listed by the order in which they
        were selected.

        The parameter ``s`` stops the selection process when there are ``s``
        items remaining and these are selected by continuing the selection,
        counting by 1 rather than by ``m``.

        Consider selecting every 3rd item from 6 until only 2 remain::

            choices    chosen
            ========   ======
              012345
              01 345   2
              01 34    25
              01  4    253
              0   4    2531
              0        25314
                       253140

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation.josephus(3, 6, 2).array_form
        [2, 5, 3, 1, 4, 0]

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Flavius_Josephus
        .. [2] https://en.wikipedia.org/wiki/Josephus_problem
        .. [3] https://web.archive.org/web/20171008094331/http://www.wou.edu/~burtonl/josephus.html

        """
    from collections import deque
    m -= 1
    Q = deque(list(range(n)))
    perm = []
    while len(Q) > max(s, 1):
        for dp in range(m):
            Q.append(Q.popleft())
        perm.append(Q.popleft())
    perm.extend(list(Q))
    return cls(perm)