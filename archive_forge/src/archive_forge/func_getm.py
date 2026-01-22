from __future__ import print_function
from copy import copy
from ..libmp.backend import xrange
def getm(fz, fb):
    m = 1 - fz / fb
    if m > 0:
        return m
    else:
        return 0.5