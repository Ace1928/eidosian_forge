import numbers
import operator
import sys
import warnings
from contextlib import suppress
from functools import reduce, wraps
from inspect import Parameter, isclass, signature
import joblib
import numpy as np
import scipy.sparse as sp
from .. import get_config as _get_config
from ..exceptions import DataConversionWarning, NotFittedError, PositiveSpectrumWarning
from ..utils._array_api import _asarray_with_order, _is_numpy_namespace, get_namespace
from ..utils.fixes import ComplexWarning, _preserve_dia_indices_dtype
from ._isfinite import FiniteStatus, cy_isfinite
from .fixes import _object_dtype_isnan
def assert_all_finite(X, *, allow_nan=False, estimator_name=None, input_name=''):
    """Throw a ValueError if X contains NaN or infinity.

    Parameters
    ----------
    X : {ndarray, sparse matrix}
        The input data.

    allow_nan : bool, default=False
        If True, do not throw error when `X` contains NaN.

    estimator_name : str, default=None
        The estimator name, used to construct the error message.

    input_name : str, default=""
        The data name used to construct the error message. In particular
        if `input_name` is "X" and the data has NaN values and
        allow_nan is False, the error message will link to the imputer
        documentation.

    Examples
    --------
    >>> from sklearn.utils import assert_all_finite
    >>> import numpy as np
    >>> array = np.array([1, np.inf, np.nan, 4])
    >>> try:
    ...     assert_all_finite(array)
    ...     print("Test passed: Array contains only finite values.")
    ... except ValueError:
    ...     print("Test failed: Array contains non-finite values.")
    Test failed: Array contains non-finite values.
    """
    _assert_all_finite(X.data if sp.issparse(X) else X, allow_nan=allow_nan, estimator_name=estimator_name, input_name=input_name)