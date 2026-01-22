import numpy as np
from ...metaarray import MetaArray
def rollingSum(data, n):
    d1 = data.copy()
    d1[1:] += d1[:-1]
    d2 = np.empty(len(d1) - n + 1, dtype=data.dtype)
    d2[0] = d1[n - 1]
    d2[1:] = d1[n:] - d1[:-n]
    return d2