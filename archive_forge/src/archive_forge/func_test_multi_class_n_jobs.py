import warnings
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import assert_almost_equal, assert_array_equal
@pytest.mark.parametrize('kernel', kernels)
def test_multi_class_n_jobs(kernel):
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X, y_mc)
    gpc_2 = GaussianProcessClassifier(kernel=kernel, n_jobs=2)
    gpc_2.fit(X, y_mc)
    y_prob = gpc.predict_proba(X2)
    y_prob_2 = gpc_2.predict_proba(X2)
    assert_almost_equal(y_prob, y_prob_2)