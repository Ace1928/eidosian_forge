from enum import Enum
from typing import Dict, List, Sequence, Tuple, cast
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.api.types import is_scalar
def is_trivial_index(index: pandas.Index) -> bool:
    """
    Check if the index is a trivial index, i.e. a sequence [0..n].

    Parameters
    ----------
    index : pandas.Index
        An index to check.

    Returns
    -------
    bool
    """
    if len(index) == 0:
        return True
    if isinstance(index, pandas.RangeIndex):
        return index.start == 0 and index.step == 1
    if not (isinstance(index, pandas.Index) and index.dtype == np.int64):
        return False
    return index.is_monotonic_increasing and index.is_unique and (index.min() == 0) and (index.max() == len(index) - 1)