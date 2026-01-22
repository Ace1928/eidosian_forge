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
Resize the permutation to the new size ``n``.

        Parameters
        ==========

        n : int
            The new size of the permutation.

        Raises
        ======

        ValueError
            If the permutation cannot be resized to the given size.
            This may only happen when resized to a smaller size than
            the original.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation

        Increasing the size of a permutation:

        >>> p = Permutation(0, 1, 2)
        >>> p = p.resize(5)
        >>> p
        (4)(0 1 2)

        Decreasing the size of the permutation:

        >>> p = p.resize(4)
        >>> p
        (3)(0 1 2)

        If resizing to the specific size breaks the cycles:

        >>> p.resize(2)
        Traceback (most recent call last):
        ...
        ValueError: The permutation cannot be resized to 2 because the
        cycle (0, 1, 2) may break.
        