from __future__ import annotations
import math
def two_factors(n: int) -> tuple[int, int]:
    """Split an integer into two integer factors.

    The two factors will be as close as possible to the sqrt of n, and are returned in decreasing
    order.  Worst case returns (n, 1).

    Args:
        n (int): The integer to factorize, must be positive.

    Return:
        tuple(int, int): The two factors of n, in decreasing order.
    """
    if n < 0:
        raise ValueError(f'two_factors expects positive integer not {n}')
    i = math.ceil(math.sqrt(n))
    while n % i != 0:
        i -= 1
    j = n // i
    if i > j:
        return (i, j)
    else:
        return (j, i)