from .libmp.backend import xrange
from .libmp import int_types, sqrt_fixed
def fracgcd(p, q):
    x, y = (p, q)
    while y:
        x, y = (y, x % y)
    if x != 1:
        p //= x
        q //= x
    if q == 1:
        return p
    return (p, q)