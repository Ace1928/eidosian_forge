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
def indexable(*iterables):
    """Make arrays indexable for cross-validation.

    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-interable objects to arrays.

    Parameters
    ----------
    *iterables : {lists, dataframes, ndarrays, sparse matrices}
        List of objects to ensure sliceability.

    Returns
    -------
    result : list of {ndarray, sparse matrix, dataframe} or None
        Returns a list containing indexable arrays (i.e. NumPy array,
        sparse matrix, or dataframe) or `None`.

    Examples
    --------
    >>> from sklearn.utils import indexable
    >>> from scipy.sparse import csr_matrix
    >>> import numpy as np
    >>> iterables = [
    ...     [1, 2, 3], np.array([2, 3, 4]), None, csr_matrix([[5], [6], [7]])
    ... ]
    >>> indexable(*iterables)
    [[1, 2, 3], array([2, 3, 4]), None, <3x1 sparse matrix ...>]
    """
    result = [_make_indexable(X) for X in iterables]
    check_consistent_length(*result)
    return result