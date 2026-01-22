import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
def test_pickleable(self):
    seq = Halton(1, scramble=False, seed=np.random.RandomState(2305982309))
    x = 3 * seq.random(50)
    xitp = 3 * seq.random(50)
    y = _1d_test_function(x)
    interp = self.build(x, y)
    yitp1 = interp(xitp)
    yitp2 = pickle.loads(pickle.dumps(interp))(xitp)
    assert_array_equal(yitp1, yitp2)