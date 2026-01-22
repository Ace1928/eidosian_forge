from functools import reduce
from operator import mul, add
def prodpow(bases, exponents):
    """
        Examples
        --------
        >>> prodpow([2, 3], [[0, 1], [1, 2]])
        [3, 18]

        """
    result = []
    for row in exponents:
        res = 1
        for b, e in zip(bases, row):
            res *= b ** e
        result.append(res)
    return result