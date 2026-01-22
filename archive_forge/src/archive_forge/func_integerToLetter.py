import io
import math
import os
import typing
import weakref
def integerToLetter(i) -> str:
    """Returns letter sequence string for integer i."""
    import string
    ls = string.ascii_uppercase
    n, a = (1, i)
    while pow(26, n) <= a:
        a -= int(math.pow(26, n))
        n += 1
    str_t = ''
    for j in reversed(range(n)):
        f, g = divmod(a, int(math.pow(26, j)))
        str_t += ls[f]
        a = g
    return str_t