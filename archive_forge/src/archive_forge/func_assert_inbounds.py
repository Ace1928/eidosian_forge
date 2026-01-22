import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
def assert_inbounds(x):
    assert np.all(x >= bnds.lb)
    assert np.all(x <= bnds.ub)