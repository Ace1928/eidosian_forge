from .sage_helper import _within_sage
from .pari import *
import re
def float_to_gen(x, precision):
    return pari._real_coerced_to_bits_prec(x, precision)