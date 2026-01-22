from sage.all import (cached_method, real_part, imag_part, round, ceil, floor, log,
import itertools
def can_express(self, a, prec=None):
    return self.express(a, prec) is not None