import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest
from statsmodels.stats.correlation_tools import (
from statsmodels.tools.testing import Holder
def norm_f(x, y):
    """Frobenious norm (squared sum) of difference between two arrays
    """
    d = ((x - y) ** 2).sum()
    return np.sqrt(d)