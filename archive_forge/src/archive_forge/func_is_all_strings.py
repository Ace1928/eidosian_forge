from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import conversion
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import ABCIndex
from pandas.core.dtypes.inference import (
def is_all_strings(value: ArrayLike) -> bool:
    """
    Check if this is an array of strings that we should try parsing.

    Includes object-dtype ndarray containing all-strings, StringArray,
    and Categorical with all-string categories.
    Does not include numpy string dtypes.
    """
    dtype = value.dtype
    if isinstance(dtype, np.dtype):
        if len(value) == 0:
            return dtype == np.dtype('object')
        else:
            return dtype == np.dtype('object') and lib.is_string_array(np.asarray(value), skipna=False)
    elif isinstance(dtype, CategoricalDtype):
        return dtype.categories.inferred_type == 'string'
    return dtype == 'string'