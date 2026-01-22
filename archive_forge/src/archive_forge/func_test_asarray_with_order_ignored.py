from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
def test_asarray_with_order_ignored():
    """Test _asarray_with_order ignores order for Generic ArrayAPI."""
    xp = pytest.importorskip('numpy.array_api')
    xp_ = _AdjustableNameAPITestWrapper(xp, 'wrapped.array_api')
    X = numpy.asarray([[1.2, 3.4, 5.1], [3.4, 5.5, 1.2]], order='C')
    X = xp_.asarray(X)
    X_new = _asarray_with_order(X, order='F', xp=xp_)
    X_new_np = numpy.asarray(X_new)
    assert X_new_np.flags['C_CONTIGUOUS']
    assert not X_new_np.flags['F_CONTIGUOUS']