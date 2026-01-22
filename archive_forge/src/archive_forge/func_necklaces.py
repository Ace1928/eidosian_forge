from collections import Counter, defaultdict, OrderedDict
from itertools import (
from itertools import product as cartes # noqa: F401
from operator import gt
from sympy.utilities.enumerative import (
from sympy.utilities.misc import as_int
from sympy.utilities.decorator import deprecated
def necklaces(n, k, free=False):
    """
    A routine to generate necklaces that may (free=True) or may not
    (free=False) be turned over to be viewed. The "necklaces" returned
    are comprised of ``n`` integers (beads) with ``k`` different
    values (colors). Only unique necklaces are returned.

    Examples
    ========

    >>> from sympy.utilities.iterables import necklaces, bracelets
    >>> def show(s, i):
    ...     return ''.join(s[j] for j in i)

    The "unrestricted necklace" is sometimes also referred to as a
    "bracelet" (an object that can be turned over, a sequence that can
    be reversed) and the term "necklace" is used to imply a sequence
    that cannot be reversed. So ACB == ABC for a bracelet (rotate and
    reverse) while the two are different for a necklace since rotation
    alone cannot make the two sequences the same.

    (mnemonic: Bracelets can be viewed Backwards, but Not Necklaces.)

    >>> B = [show('ABC', i) for i in bracelets(3, 3)]
    >>> N = [show('ABC', i) for i in necklaces(3, 3)]
    >>> set(N) - set(B)
    {'ACB'}

    >>> list(necklaces(4, 2))
    [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1),
     (0, 1, 0, 1), (0, 1, 1, 1), (1, 1, 1, 1)]

    >>> [show('.o', i) for i in bracelets(4, 2)]
    ['....', '...o', '..oo', '.o.o', '.ooo', 'oooo']

    References
    ==========

    .. [1] https://mathworld.wolfram.com/Necklace.html

    .. [2] Frank Ruskey, Carla Savage, and Terry Min Yih Wang,
        Generating necklaces, Journal of Algorithms 13 (1992), 414-430;
        https://doi.org/10.1016/0196-6774(92)90047-G

    """
    if k == 0 and n > 0:
        return
    a = [0] * n
    yield tuple(a)
    if n == 0:
        return
    while True:
        i = n - 1
        while a[i] == k - 1:
            i -= 1
            if i == -1:
                return
        a[i] += 1
        for j in range(n - i - 1):
            a[j + i + 1] = a[j]
        if n % (i + 1) == 0 and (not free or all((a <= a[j::-1] + a[-1:j:-1] for j in range(n - 1)))):
            yield tuple(a)