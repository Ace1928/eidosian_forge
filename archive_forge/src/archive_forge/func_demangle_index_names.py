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
@staticmethod
def demangle_index_names(cols: List[str]) -> Union[_COL_NAME_TYPE, List[_COL_NAME_TYPE]]:
    """
        Demangle index column names to index labels.

        Parameters
        ----------
        cols : list of str
            Index column names.

        Returns
        -------
        list or a single demangled name
            Demangled index names.
        """
    if len(cols) == 1:
        return ColNameCodec.demangle_index_name(cols[0])
    return [ColNameCodec.demangle_index_name(n) for n in cols]