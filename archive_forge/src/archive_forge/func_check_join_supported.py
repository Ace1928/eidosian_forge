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
def check_join_supported(join_type: str):
    """
    Check if join type is supported by HDK.

    Parameters
    ----------
    join_type : str
        Join type.

    Returns
    -------
    None
    """
    if join_type not in ('inner', 'left'):
        raise NotImplementedError(f'{join_type} join')