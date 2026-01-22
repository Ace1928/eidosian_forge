from bisect import bisect
from ..libmp.backend import xrange
def mpolyval(ser, a):
    return [ctx.polyval(s[::-1], a) for s in ser]