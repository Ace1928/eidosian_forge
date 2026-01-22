import math
from bisect import bisect
from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO
def list_primes(n):
    return [int(_) for _ in sage.primes(n + 1)]