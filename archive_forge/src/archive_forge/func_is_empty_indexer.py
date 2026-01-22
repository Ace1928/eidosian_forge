from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
def is_empty_indexer(indexer) -> bool:
    """
    Check if we have an empty indexer.

    Parameters
    ----------
    indexer : object

    Returns
    -------
    bool
    """
    if is_list_like(indexer) and (not len(indexer)):
        return True
    if not isinstance(indexer, tuple):
        indexer = (indexer,)
    return any((isinstance(idx, np.ndarray) and len(idx) == 0 for idx in indexer))