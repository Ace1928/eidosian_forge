import pickle
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from .test_continuous_basic import check_distribution_rvs
import numpy
import numpy as np
import scipy.linalg
from scipy.stats._multivariate import (_PSD,
from scipy.stats import (multivariate_normal, multivariate_hypergeom,
from scipy.stats import _covariance, Covariance
from scipy import stats
from scipy.integrate import romb, qmc_quad, tplquad
from scipy.special import multigammaln
from scipy._lib._pep440 import Version
from .common_tests import check_random_state_property
from .data._mvt import _qsimvtv
from unittest.mock import patch
def test_default_inputs(self):
    num_rows = 4
    num_cols = 3
    M = np.full((num_rows, num_cols), 0.3)
    U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
    V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)
    Z = np.zeros((num_rows, num_cols))
    Zr = np.zeros((num_rows, 1))
    Zc = np.zeros((1, num_cols))
    Ir = np.identity(num_rows)
    Ic = np.identity(num_cols)
    I1 = np.identity(1)
    assert_equal(matrix_normal.rvs(mean=M, rowcov=U, colcov=V).shape, (num_rows, num_cols))
    assert_equal(matrix_normal.rvs(mean=M).shape, (num_rows, num_cols))
    assert_equal(matrix_normal.rvs(rowcov=U).shape, (num_rows, 1))
    assert_equal(matrix_normal.rvs(colcov=V).shape, (1, num_cols))
    assert_equal(matrix_normal.rvs(mean=M, colcov=V).shape, (num_rows, num_cols))
    assert_equal(matrix_normal.rvs(mean=M, rowcov=U).shape, (num_rows, num_cols))
    assert_equal(matrix_normal.rvs(rowcov=U, colcov=V).shape, (num_rows, num_cols))
    assert_equal(matrix_normal(mean=M).rowcov, Ir)
    assert_equal(matrix_normal(mean=M).colcov, Ic)
    assert_equal(matrix_normal(rowcov=U).mean, Zr)
    assert_equal(matrix_normal(rowcov=U).colcov, I1)
    assert_equal(matrix_normal(colcov=V).mean, Zc)
    assert_equal(matrix_normal(colcov=V).rowcov, I1)
    assert_equal(matrix_normal(mean=M, rowcov=U).colcov, Ic)
    assert_equal(matrix_normal(mean=M, colcov=V).rowcov, Ir)
    assert_equal(matrix_normal(rowcov=U, colcov=V).mean, Z)