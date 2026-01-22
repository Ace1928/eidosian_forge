import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_label_structuring_elements():
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'label_inputs.txt'))
    strels = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'label_strels.txt'))
    results = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'label_results.txt'))
    data = data.reshape((-1, 7, 7))
    strels = strels.reshape((-1, 3, 3))
    results = results.reshape((-1, 7, 7))
    r = 0
    for i in range(data.shape[0]):
        d = data[i, :, :]
        for j in range(strels.shape[0]):
            s = strels[j, :, :]
            assert_equal(ndimage.label(d, s)[0], results[r, :, :])
            r += 1