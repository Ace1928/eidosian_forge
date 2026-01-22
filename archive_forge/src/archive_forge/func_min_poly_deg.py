from sage.all import (cached_method, real_part, imag_part, round, ceil, floor, log,
import itertools
def min_poly_deg(z):
    return z.min_polynomial().degree()