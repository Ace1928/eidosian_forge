import random
from bisect import bisect
from itertools import count
from array import array as _array
from sympy.core.function import Function
from sympy.core.singleton import S
from .primetest import isprime
from sympy.utilities.misc import as_int
def totientrange(self, a, b):
    """Generate all totient numbers for the range [a, b).

        Examples
        ========

        >>> from sympy import sieve
        >>> print([i for i in sieve.totientrange(7, 18)])
        [6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16]
        """
    a = max(1, _as_int_ceiling(a))
    b = _as_int_ceiling(b)
    n = len(self._tlist)
    if a >= b:
        return
    elif b <= n:
        for i in range(a, b):
            yield self._tlist[i]
    else:
        self._tlist += _arange(n, b)
        for i in range(1, n):
            ti = self._tlist[i]
            startindex = (n + i - 1) // i * i
            for j in range(startindex, b, i):
                self._tlist[j] -= ti
            if i >= a:
                yield ti
        for i in range(n, b):
            ti = self._tlist[i]
            for j in range(2 * i, b, i):
                self._tlist[j] -= ti
            if i >= a:
                yield ti