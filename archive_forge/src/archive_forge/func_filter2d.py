import numpy as np
from numpy.testing import assert_allclose
from scipy import ndimage
from scipy.ndimage import _ctest
from scipy.ndimage import _cytest
from scipy._lib._ccallback import LowLevelCallable
def filter2d(footprint_elements, weights):
    return (weights * footprint_elements).sum()