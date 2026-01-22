import math
import numbers
import platform
import struct
import timeit
import warnings
from collections.abc import Sequence
from contextlib import contextmanager, suppress
from itertools import compress, islice
import numpy as np
from scipy.sparse import issparse
from .. import get_config
from ..exceptions import DataConversionWarning
from . import _joblib, metadata_routing
from ._bunch import Bunch
from ._estimator_html_repr import estimator_html_repr
from ._param_validation import Integral, Interval, validate_params
from .class_weight import compute_class_weight, compute_sample_weight
from .deprecation import deprecated
from .discovery import all_estimators
from .fixes import parse_version, threadpool_info
from .murmurhash import murmurhash3_32
from .validation import (
def tosequence(x):
    """Cast iterable x to a Sequence, avoiding a copy if possible.

    Parameters
    ----------
    x : iterable
        The iterable to be converted.

    Returns
    -------
    x : Sequence
        If `x` is a NumPy array, it returns it as a `ndarray`. If `x`
        is a `Sequence`, `x` is returned as-is. If `x` is from any other
        type, `x` is returned casted as a list.
    """
    if isinstance(x, np.ndarray):
        return np.asarray(x)
    elif isinstance(x, Sequence):
        return x
    else:
        return list(x)