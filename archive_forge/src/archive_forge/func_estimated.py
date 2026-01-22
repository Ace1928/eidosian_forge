from functools import partial
from itertools import product
import numpy as np
import pytest
from numpy.testing import (assert_allclose, assert_, assert_equal,
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import aslinearoperator
import scipy.linalg
from scipy.sparse.linalg import expm as sp_expm
from scipy.sparse.linalg._expm_multiply import (_theta, _compute_p_max,
from scipy._lib._util import np_long
def estimated(func):
    """If trace is estimated, it should warn.

    We warn that estimation of trace might impact performance.
    All result have to be correct nevertheless!

    """

    def wrapped(*args, **kwds):
        with pytest.warns(UserWarning, match='Trace of LinearOperator not available'):
            return func(*args, **kwds)
    return wrapped