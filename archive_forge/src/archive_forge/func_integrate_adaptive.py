import math
import warnings
import numpy as np
from .util import import_
@staticmethod
def integrate_adaptive(rhs, jac, y0, x0, xend, dx0, **kwargs):
    if kwargs:
        warnings.warn('Ignoring keyword-argumtents: %s' % ', '.join(kwargs.keys()))
    xspan = xend - x0
    n = int(math.ceil(xspan / dx0))
    yout = [y0[:]]
    xout = [x0]
    k = [np.empty(len(y0)) for _ in range(4)]
    for i in range(0, n + 1):
        x, y = (xout[-1], yout[-1])
        h = min(dx0, xend - x)
        rhs(x, y, k[0])
        rhs(x + h / 2, y + h / 2 * k[0], k[1])
        rhs(x + h / 2, y + h / 2 * k[1], k[2])
        rhs(x + h, y + h * k[2], k[3])
        yout.append(y + h / 6 * (k[0] + 2 * k[1] + 2 * k[2] + k[3]))
        xout.append(x + h)
    return (np.array(xout), np.array(yout), {'nfev': n * 4})