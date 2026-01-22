from __future__ import (absolute_import, division, print_function)
import math
def isnotanumber(x):
    try:
        return math.isnan(x)
    except TypeError:
        return False