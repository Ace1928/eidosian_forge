import operator
import math
import cmath
def tanpi(x):
    try:
        return sinpi(x) / cospi(x)
    except OverflowError:
        if complex(x).imag > 10:
            return 1j
        if complex(x).imag < 10:
            return -1j
        raise