from __future__ import annotations
from collections import abc
from typing import (
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core import (
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
from pandas.core.indexes.api import (
from pandas.core.internals.array_manager import (
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def mgr_to_mgr(mgr, typ: str, copy: bool=True) -> Manager:
    """
    Convert to specific type of Manager. Does not copy if the type is already
    correct. Does not guarantee a copy otherwise. `copy` keyword only controls
    whether conversion from Block->ArrayManager copies the 1D arrays.
    """
    new_mgr: Manager
    if typ == 'block':
        if isinstance(mgr, BlockManager):
            new_mgr = mgr
        elif mgr.ndim == 2:
            new_mgr = arrays_to_mgr(mgr.arrays, mgr.axes[0], mgr.axes[1], typ='block')
        else:
            new_mgr = SingleBlockManager.from_array(mgr.arrays[0], mgr.index)
    elif typ == 'array':
        if isinstance(mgr, ArrayManager):
            new_mgr = mgr
        elif mgr.ndim == 2:
            arrays = [mgr.iget_values(i) for i in range(len(mgr.axes[0]))]
            if copy:
                arrays = [arr.copy() for arr in arrays]
            new_mgr = ArrayManager(arrays, [mgr.axes[1], mgr.axes[0]])
        else:
            array = mgr.internal_values()
            if copy:
                array = array.copy()
            new_mgr = SingleArrayManager([array], [mgr.index])
    else:
        raise ValueError(f"'typ' needs to be one of {{'block', 'array'}}, got '{typ}'")
    return new_mgr