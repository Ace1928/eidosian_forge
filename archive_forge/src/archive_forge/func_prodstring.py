from .libmp.backend import xrange
from .libmp import int_types, sqrt_fixed
def prodstring(r, constants):
    q = r[0]
    r = r[1:]
    num = []
    den = []
    for i in range(len(r)):
        p = r[i]
        if p:
            z = fracgcd(-p, q)
            cs = constants[i][1]
            if isinstance(z, int_types):
                if abs(z) == 1:
                    t = cs
                else:
                    t = '%s**%s' % (cs, abs(z))
                [num, den][z < 0].append(t)
            else:
                t = '%s**(%s/%s)' % (cs, abs(z[0]), z[1])
                [num, den][z[0] < 0].append(t)
    num = '*'.join(num)
    den = '*'.join(den)
    if num and den:
        return '(%s)/(%s)' % (num, den)
    if num:
        return num
    if den:
        return '1/(%s)' % den