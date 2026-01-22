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
def compute_sliced_len(slc, sequence_len):
    """
    Compute length of sliced object.

    Parameters
    ----------
    slc : slice
        Slice object.
    sequence_len : int
        Length of sequence, to which slice will be applied.

    Returns
    -------
    int
        Length of object after applying slice object on it.
    """
    return len(range(*slc.indices(sequence_len)))