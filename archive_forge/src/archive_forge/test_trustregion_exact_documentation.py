import numpy as np
from scipy.optimize._trustregion_exact import (
from scipy.linalg import (svd, get_lapack_funcs, det, qr, norm)
from numpy.testing import (assert_array_equal,

Unit tests for trust-region iterative subproblem.

To run it in its simplest form::
  nosetests test_optimize.py

