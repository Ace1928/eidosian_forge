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
def test_bound_check_fixed_hyperparameter():
    k1 = 50.0 ** 2 * RBF(length_scale=50.0)
    k2 = ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds='fixed')
    kernel = k1 + k2
    GaussianProcessRegressor(kernel=kernel).fit(X, y)