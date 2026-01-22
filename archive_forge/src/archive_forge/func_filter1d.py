import numpy as np
from numpy.testing import assert_allclose
from scipy import ndimage
from scipy.ndimage import _ctest
from scipy.ndimage import _cytest
from scipy._lib._ccallback import LowLevelCallable
def filter1d(input_line, output_line, filter_size):
    for i in range(output_line.size):
        output_line[i] = 0
        for j in range(filter_size):
            output_line[i] += input_line[i + j]
    output_line /= filter_size