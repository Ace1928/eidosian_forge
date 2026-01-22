from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import is_list_like
from pandas.core.arrays import ExtensionArray
def putmask_without_repeat(values: np.ndarray, mask: npt.NDArray[np.bool_], new: Any) -> None:
    """
    np.putmask will truncate or repeat if `new` is a listlike with
    len(new) != len(values).  We require an exact match.

    Parameters
    ----------
    values : np.ndarray
    mask : np.ndarray[bool]
    new : Any
    """
    if getattr(new, 'ndim', 0) >= 1:
        new = new.astype(values.dtype, copy=False)
    nlocs = mask.sum()
    if nlocs > 0 and is_list_like(new) and (getattr(new, 'ndim', 1) == 1):
        shape = np.shape(new)
        if nlocs == shape[-1]:
            np.place(values, mask, new)
        elif mask.shape[-1] == shape[-1] or shape[-1] == 1:
            np.putmask(values, mask, new)
        else:
            raise ValueError('cannot assign mismatch length to masked array')
    else:
        np.putmask(values, mask, new)