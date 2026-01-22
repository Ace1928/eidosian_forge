import numpy as np
from matplotlib import cbook, units
import matplotlib.projections.polar as polar
def rad_fn(x, pos=None):
    """Radian function formatter."""
    n = int(x / np.pi * 2.0 + 0.25)
    if n == 0:
        return str(x)
    elif n == 1:
        return '$\\pi/2$'
    elif n == 2:
        return '$\\pi$'
    elif n % 2 == 0:
        return f'${n // 2}\\pi$'
    else:
        return f'${n}\\pi/2$'