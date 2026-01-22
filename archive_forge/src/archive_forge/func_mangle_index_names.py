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
def mangle_index_names(names: List[_COL_NAME_TYPE]) -> List[str]:
    """
        Return mangled index names for index labels.

        Mangled names are used for index columns because index
        labels cannot always be used as HDK table column
        names. E.e. label can be a non-string value or an
        unallowed string (empty strings, etc.) for a table column
        name.

        Parameters
        ----------
        names : list of str
            Index labels.

        Returns
        -------
        list of str
            Mangled names.
        """
    pref = ColNameCodec.IDX_COL_NAME
    return [f'{pref}{i}_{ColNameCodec.encode(n)}' for i, n in enumerate(names)]