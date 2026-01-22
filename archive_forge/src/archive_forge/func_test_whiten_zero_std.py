import warnings
import sys
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.cluster.vq import (kmeans, kmeans2, py_vq, vq, whiten,
from scipy.cluster import _vq
from scipy.conftest import (
from scipy.sparse._sputils import matrix
from scipy._lib._array_api import (
@array_api_compatible
def test_whiten_zero_std(self, xp):
    desired = xp.asarray([[0.0, 1.0, 2.86666544], [0.0, 1.0, 1.32460034], [0.0, 1.0, 3.74382172]])
    obs = xp.asarray([[0.0, 1.0, 0.74109533], [0.0, 1.0, 0.34243798], [0.0, 1.0, 0.96785929]])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        xp_assert_close(whiten(obs), desired, rtol=1e-05)
        assert_equal(len(w), 1)
        assert_(issubclass(w[-1].category, RuntimeWarning))