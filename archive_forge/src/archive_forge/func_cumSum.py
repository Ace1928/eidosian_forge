from fontTools.ttLib import TTFont
from collections import defaultdict
from operator import add
from functools import reduce
def cumSum(f, op=add, start=0, decreasing=False):
    keys = sorted(f.keys())
    minx, maxx = (keys[0], keys[-1])
    total = reduce(op, f.values(), start)
    if decreasing:
        missing = lambda x: start if x > maxx else total
        domain = range(maxx, minx - 1, -1)
    else:
        missing = lambda x: start if x < minx else total
        domain = range(minx, maxx + 1)
    out = missingdict(missing)
    v = start
    for x in domain:
        v = op(v, f[x])
        out[x] = v
    return out