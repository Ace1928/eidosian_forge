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
def test_check_preserve_type():
    XA = np.resize(np.arange(40), (5, 8)).astype(np.float32)
    XB = np.resize(np.arange(40), (5, 8)).astype(np.float32)
    XA_checked, XB_checked = check_pairwise_arrays(XA, None)
    assert XA_checked.dtype == np.float32
    XA_checked, XB_checked = check_pairwise_arrays(XA, XB)
    assert XA_checked.dtype == np.float32
    assert XB_checked.dtype == np.float32
    XA_checked, XB_checked = check_pairwise_arrays(XA.astype(float), XB)
    assert XA_checked.dtype == float
    assert XB_checked.dtype == float
    XA_checked, XB_checked = check_pairwise_arrays(XA, XB.astype(float))
    assert XA_checked.dtype == float
    assert XB_checked.dtype == float