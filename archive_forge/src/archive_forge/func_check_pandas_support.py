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
def check_pandas_support(caller_name):
    """Raise ImportError with detailed error message if pandas is not installed.

    Plot utilities like :func:`fetch_openml` should lazily import
    pandas and call this helper before any computation.

    Parameters
    ----------
    caller_name : str
        The name of the caller that requires pandas.

    Returns
    -------
    pandas
        The pandas package.
    """
    try:
        import pandas
        return pandas
    except ImportError as e:
        raise ImportError('{} requires pandas.'.format(caller_name)) from e