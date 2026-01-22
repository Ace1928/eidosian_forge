from greenlet import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def gr2(n, seen):
    for ii in gr1(n):
        seen.append(ii)