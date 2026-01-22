from .sage_helper import _within_sage
from .pari import *
import re
def complex_to_gen(x, precision):
    return pari.complex(pari._real_coerced_to_bits_prec(x.real, precision), pari._real_coerced_to_bits_prec(x.imag, precision))