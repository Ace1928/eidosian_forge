from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Union
import numpy as np
import pandas
from pandas.api.types import is_bool, is_list_like
from pandas.core.dtypes.common import is_bool_dtype, is_integer, is_integer_dtype
from pandas.core.indexing import IndexingError
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from .dataframe import DataFrame
from .series import Series
from .utils import is_scalar
def is_slice(x):
    """
    Check that argument is an instance of slice.

    Parameters
    ----------
    x : object
        Object to check.

    Returns
    -------
    bool
        True if argument is a slice, False otherwise.
    """
    return isinstance(x, slice)