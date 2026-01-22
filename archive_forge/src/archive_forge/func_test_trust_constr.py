import logging
import sys
import numpy
import numpy as np
import time
from multiprocessing import Pool
from numpy.testing import assert_allclose, IS_PYPY
import pytest
from pytest import raises as assert_raises, warns
from scipy.optimize import (shgo, Bounds, minimize_scalar, minimize, rosen,
from scipy.optimize._constraints import new_constraint_to_old
from scipy.optimize._shgo import SHGO
@pytest.mark.filterwarnings('ignore:delta_grad')
def test_trust_constr():

    def quad(x):
        x = np.asarray(x)
        return [np.sum(x ** 2)]
    nlc = NonlinearConstraint(quad, [2.6], [3])
    minimizer_kwargs = {'method': 'trust-constr'}
    res = shgo(rosen, [(0, 10), (0, 10)], constraints=nlc, sampling_method='sobol', minimizer_kwargs=minimizer_kwargs)
    assert np.all(np.sum(res.x ** 2) >= 2.6)
    assert np.all(np.sum(res.x ** 2) <= 3.0)
    assert res.success