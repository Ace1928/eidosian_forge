import random
from bisect import bisect
from itertools import count
from array import array as _array
from sympy.core.function import Function
from sympy.core.singleton import S
from .primetest import isprime
from sympy.utilities.misc import as_int
def cycle_length(f, x0, nmax=None, values=False):
    """For a given iterated sequence, return a generator that gives
    the length of the iterated cycle (lambda) and the length of terms
    before the cycle begins (mu); if ``values`` is True then the
    terms of the sequence will be returned instead. The sequence is
    started with value ``x0``.

    Note: more than the first lambda + mu terms may be returned and this
    is the cost of cycle detection with Brent's method; there are, however,
    generally less terms calculated than would have been calculated if the
    proper ending point were determined, e.g. by using Floyd's method.

    >>> from sympy.ntheory.generate import cycle_length

    This will yield successive values of i <-- func(i):

        >>> def iter(func, i):
        ...     while 1:
        ...         ii = func(i)
        ...         yield ii
        ...         i = ii
        ...

    A function is defined:

        >>> func = lambda i: (i**2 + 1) % 51

    and given a seed of 4 and the mu and lambda terms calculated:

        >>> next(cycle_length(func, 4))
        (6, 2)

    We can see what is meant by looking at the output:

        >>> n = cycle_length(func, 4, values=True)
        >>> list(ni for ni in n)
        [17, 35, 2, 5, 26, 14, 44, 50, 2, 5, 26, 14]

    There are 6 repeating values after the first 2.

    If a sequence is suspected of being longer than you might wish, ``nmax``
    can be used to exit early (and mu will be returned as None):

        >>> next(cycle_length(func, 4, nmax = 4))
        (4, None)
        >>> [ni for ni in cycle_length(func, 4, nmax = 4, values=True)]
        [17, 35, 2, 5]

    Code modified from:
        https://en.wikipedia.org/wiki/Cycle_detection.
    """
    nmax = int(nmax or 0)
    power = lam = 1
    tortoise, hare = (x0, f(x0))
    i = 0
    while tortoise != hare and (not nmax or i < nmax):
        i += 1
        if power == lam:
            tortoise = hare
            power *= 2
            lam = 0
        if values:
            yield hare
        hare = f(hare)
        lam += 1
    if nmax and i == nmax:
        if values:
            return
        else:
            yield (nmax, None)
            return
    if not values:
        mu = 0
        tortoise = hare = x0
        for i in range(lam):
            hare = f(hare)
        while tortoise != hare:
            tortoise = f(tortoise)
            hare = f(hare)
            mu += 1
        if mu:
            mu -= 1
        yield (lam, mu)