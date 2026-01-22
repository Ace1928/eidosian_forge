from ..libmp.backend import xrange
import math
import cmath
def defun(f):
    SpecialFunctions.defined_functions[f.__name__] = (f, False)
    return f