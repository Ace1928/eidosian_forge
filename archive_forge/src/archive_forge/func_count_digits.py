from collections import defaultdict
from sympy.utilities.iterables import multiset, is_palindromic as _palindromic
from sympy.utilities.misc import as_int
def count_digits(n, b=10):
    """
    Return a dictionary whose keys are the digits of ``n`` in the
    given base, ``b``, with keys indicating the digits appearing in the
    number and values indicating how many times that digit appeared.

    Examples
    ========

    >>> from sympy.ntheory import count_digits

    >>> count_digits(1111339)
    {1: 4, 3: 2, 9: 1}

    The digits returned are always represented in base-10
    but the number itself can be entered in any format that is
    understood by Python; the base of the number can also be
    given if it is different than 10:

    >>> n = 0xFA; n
    250
    >>> count_digits(_)
    {0: 1, 2: 1, 5: 1}
    >>> count_digits(n, 16)
    {10: 1, 15: 1}

    The default dictionary will return a 0 for any digit that did
    not appear in the number. For example, which digits appear 7
    times in ``77!``:

    >>> from sympy import factorial
    >>> c77 = count_digits(factorial(77))
    >>> [i for i in range(10) if c77[i] == 7]
    [1, 3, 7, 9]
    """
    rv = defaultdict(int, multiset(digits(n, b)).items())
    rv.pop(b) if b in rv else rv.pop(-b)
    return rv