from greenlet import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def gr1(n):
    for ii in range(1, n):
        Yield(ii)
        Yield(ii * ii, 2)