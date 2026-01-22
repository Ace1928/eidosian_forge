from .sage_helper import _within_sage
from .pari import *
import re
def is_exact(x):
    if isinstance(x, int):
        return True
    if isinstance(x, Gen):
        return x.precision() == precision_of_exact_GEN
    if isinstance(x, Number):
        return x.gen.precision() == precision_of_exact_GEN
    return False