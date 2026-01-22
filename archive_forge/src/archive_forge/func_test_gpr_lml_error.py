import re
import sys
import warnings
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import (
def test_gpr_lml_error():
    """Check that we raise the proper error in the LML method."""
    gpr = GaussianProcessRegressor(kernel=RBF()).fit(X, y)
    err_msg = 'Gradient can only be evaluated for theta!=None'
    with pytest.raises(ValueError, match=err_msg):
        gpr.log_marginal_likelihood(eval_gradient=True)