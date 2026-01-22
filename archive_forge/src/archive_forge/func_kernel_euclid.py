import numpy as np
from scipy import spatial as ssp
import matplotlib.pylab as plt
def kernel_euclid(x, y, p=2, **kwds):
    return ssp.minkowski_distance(x[:, np.newaxis, :], y[np.newaxis, :, :], p)