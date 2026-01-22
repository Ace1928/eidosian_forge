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
def check_memory(memory):
    """Check that ``memory`` is joblib.Memory-like.

    joblib.Memory-like means that ``memory`` can be converted into a
    joblib.Memory instance (typically a str denoting the ``location``)
    or has the same interface (has a ``cache`` method).

    Parameters
    ----------
    memory : None, str or object with the joblib.Memory interface
        - If string, the location where to create the `joblib.Memory` interface.
        - If None, no caching is done and the Memory object is completely transparent.

    Returns
    -------
    memory : object with the joblib.Memory interface
        A correct joblib.Memory object.

    Raises
    ------
    ValueError
        If ``memory`` is not joblib.Memory-like.

    Examples
    --------
    >>> from sklearn.utils.validation import check_memory
    >>> check_memory("caching_dir")
    Memory(location=caching_dir/joblib)
    """
    if memory is None or isinstance(memory, str):
        memory = joblib.Memory(location=memory, verbose=0)
    elif not hasattr(memory, 'cache'):
        raise ValueError("'memory' should be None, a string or have the same interface as joblib.Memory. Got memory='{}' instead.".format(memory))
    return memory