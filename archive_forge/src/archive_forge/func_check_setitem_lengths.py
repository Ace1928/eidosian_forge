from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
def check_setitem_lengths(indexer, value, values) -> bool:
    """
    Validate that value and indexer are the same length.

    An special-case is allowed for when the indexer is a boolean array
    and the number of true values equals the length of ``value``. In
    this case, no exception is raised.

    Parameters
    ----------
    indexer : sequence
        Key for the setitem.
    value : array-like
        Value for the setitem.
    values : array-like
        Values being set into.

    Returns
    -------
    bool
        Whether this is an empty listlike setting which is a no-op.

    Raises
    ------
    ValueError
        When the indexer is an ndarray or list and the lengths don't match.
    """
    no_op = False
    if isinstance(indexer, (np.ndarray, list)):
        if is_list_like(value):
            if len(indexer) != len(value) and values.ndim == 1:
                if isinstance(indexer, list):
                    indexer = np.array(indexer)
                if not (isinstance(indexer, np.ndarray) and indexer.dtype == np.bool_ and (indexer.sum() == len(value))):
                    raise ValueError('cannot set using a list-like indexer with a different length than the value')
            if not len(indexer):
                no_op = True
    elif isinstance(indexer, slice):
        if is_list_like(value):
            if len(value) != length_of_indexer(indexer, values) and values.ndim == 1:
                raise ValueError('cannot set using a slice indexer with a different length than the value')
            if not len(value):
                no_op = True
    return no_op