import unittest
import numpy as np
import scipy.linalg
from skimage import data, img_as_float
from pygsp import graphs
def correct_sign(U):
    signs = np.sign(U[0, :])
    signs[signs == 0] = 1
    return U * signs