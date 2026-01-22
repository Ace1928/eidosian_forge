from os.path import dirname
from os.path import join
import numpy as np
import scipy.fftpack
import scipy.io
import scipy.misc
import scipy.ndimage
import scipy.stats
from ..motion import blockMotion
from ..utils import *
def zigzag(data):
    nrows, ncols = data.shape
    d = sum([list(data[::-1, :].diagonal(i)[::(i + nrows + 1) % 2 * -2 + 1]) for i in range(-nrows, nrows + len(data[0]))], [])
    return np.array(d)