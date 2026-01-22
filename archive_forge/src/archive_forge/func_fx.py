import numpy as np
from . import Filter  # prevent circular import in Python < 3.5
def fx(x, a):
    y = np.exp(-float(a) / x)
    if isinstance(x, np.ndarray):
        y = np.where(x <= 0, 0.0, y)
    elif x <= 0:
        y = 0.0
    return y