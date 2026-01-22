from sympy.core import Basic, Integer
import random

        Unranks an n-bit sized Gray code of rank k. This method exists
        so that a derivative GrayCode class can define its own code of
        a given rank.

        The string here is generated in reverse order to allow for tail-call
        optimization.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> GrayCode(5, rank=3).current
        '00010'
        >>> GrayCode.unrank(5, 3)
        '00010'

        See Also
        ========

        rank
        