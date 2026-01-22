from sympy.core import Integer, Pow, Mod
from sympy import factorint
def is_cyclic_number(n):
    """
    Check whether `n` is a cyclic number. A number `n` is said to be cyclic
    if and only if every finite group of order `n` is cyclic. For more
    information see [1]_.

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import is_cyclic_number
    >>> from sympy import randprime
    >>> is_cyclic_number(15)
    True
    >>> is_cyclic_number(randprime(1, 2000)**2)
    False
    >>> is_cyclic_number(4)
    False

    References
    ==========

    .. [1] Pakianathan, J., Shankar, K., *Nilpotent Numbers*,
            The American Mathematical Monthly, 107(7), 631-634.

    """
    if n <= 0 or int(n) != n:
        raise ValueError('n must be a positive integer, not %i' % n)
    n = Integer(n)
    if not is_nilpotent_number(n):
        return False
    prime_factors = list(factorint(n).items())
    is_cyclic = all((a_i < 2 for p_i, a_i in prime_factors))
    return is_cyclic