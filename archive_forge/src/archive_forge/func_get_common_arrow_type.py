from __future__ import annotations
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType
from pandas.core.dtypes.common import _get_dtype, is_string_dtype
from pyarrow.types import is_dictionary
from modin.pandas.indexing import is_range_like
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
def get_common_arrow_type(t1: pa.lib.DataType, t2: pa.lib.DataType) -> pa.lib.DataType:
    """
    Get common arrow data type.

    Parameters
    ----------
    t1 : pa.lib.DataType
    t2 : pa.lib.DataType

    Returns
    -------
    pa.lib.DataType
    """
    if t1 == t2:
        return t1
    if pa.types.is_string(t1):
        return t1
    if pa.types.is_string(t2):
        return t2
    if pa.types.is_null(t1):
        return t2
    if pa.types.is_null(t2):
        return t1
    t1 = t1.to_pandas_dtype()
    t2 = t2.to_pandas_dtype()
    return pa.from_numpy_dtype(np.promote_types(t1, t2))