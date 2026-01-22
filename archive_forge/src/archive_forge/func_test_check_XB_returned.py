import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
def test_check_XB_returned():
    XA = np.resize(np.arange(40), (5, 8))
    XB = np.resize(np.arange(32), (4, 8))
    XA_checked, XB_checked = check_pairwise_arrays(XA, XB)
    assert_array_equal(XA, XA_checked)
    assert_array_equal(XB, XB_checked)
    XB = np.resize(np.arange(40), (5, 8))
    XA_checked, XB_checked = check_paired_arrays(XA, XB)
    assert_array_equal(XA, XA_checked)
    assert_array_equal(XB, XB_checked)