from mpmath.libmp import *
from mpmath import mpf, mp
from random import randint, choice, seed
def test_tight_integer_division():
    N = 100
    seed(1)
    for i in range(N):
        a = choice([1, -1]) * randint(1, 1 << randint(10, 100))
        b = choice([1, -1]) * randint(1, 1 << randint(10, 100))
        p = a * b
        width = bitcount(abs(b)) - trailing(b)
        a = fi(a)
        b = fi(b)
        p = fi(p)
        for mode in all_modes:
            assert mpf_div(p, a, width, mode) == b