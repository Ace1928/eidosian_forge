from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import is_list_like
from pandas.core.arrays import ExtensionArray
def extract_bool_array(mask: ArrayLike) -> npt.NDArray[np.bool_]:
    """
    If we have a SparseArray or BooleanArray, convert it to ndarray[bool].
    """
    if isinstance(mask, ExtensionArray):
        mask = mask.to_numpy(dtype=bool, na_value=False)
    mask = np.asarray(mask, dtype=bool)
    return mask