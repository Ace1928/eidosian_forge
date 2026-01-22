import numpy as np
import os
import scipy.ndimage
import scipy.spatial
from ..utils import *
def gauss_window(lw, sigma):
    sd = float(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    ss = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        ss += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= ss
    return weights