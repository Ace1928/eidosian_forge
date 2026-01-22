from numpy.testing import assert_equal
import numpy as np
def signstr(x, noplus=False):
    if x in [-1, 0, 1]:
        if not noplus:
            return '+' if np.sign(x) >= 0 else '-'
        else:
            return '' if np.sign(x) >= 0 else '-'
    else:
        return str(x)