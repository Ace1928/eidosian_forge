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
def check_non_negative(X, whom):
    """
    Check if there is any negative value in an array.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        Input data.

    whom : str
        Who passed X to this function.
    """
    xp, _ = get_namespace(X)
    if sp.issparse(X):
        if X.format in ['lil', 'dok']:
            X = X.tocsr()
        if X.data.size == 0:
            X_min = 0
        else:
            X_min = X.data.min()
    else:
        X_min = xp.min(X)
    if X_min < 0:
        raise ValueError('Negative values in data passed to %s' % whom)