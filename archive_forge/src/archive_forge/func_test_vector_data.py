import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
def test_vector_data(self):
    seq = Halton(2, scramble=False, seed=np.random.RandomState())
    x = seq.random(100)
    xitp = seq.random(100)
    y = np.array([_2d_test_function(x), _2d_test_function(x[:, ::-1])]).T
    yitp1 = self.build(x, y)(xitp)
    yitp2 = self.build(x, y[:, 0])(xitp)
    yitp3 = self.build(x, y[:, 1])(xitp)
    assert_allclose(yitp1[:, 0], yitp2)
    assert_allclose(yitp1[:, 1], yitp3)