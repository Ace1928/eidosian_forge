import numpy as np
from scipy import spatial as ssp
import matplotlib.pylab as plt
def kernel_rbf(x, y, scale=1, **kwds):
    dist = ssp.minkowski_distance_p(x[:, np.newaxis, :], y[np.newaxis, :, :], 2)
    return np.exp(-0.5 / scale * dist)