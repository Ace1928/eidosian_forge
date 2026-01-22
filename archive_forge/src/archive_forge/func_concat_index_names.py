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
def concat_index_names(frames) -> Dict[str, Any]:
    """
        Calculate the index names and dtypes.

        Calculate the index names and dtypes, that the index
        columns will have after the frames concatenation.

        Parameters
        ----------
        frames : list[HdkOnNativeDataframe]

        Returns
        -------
        Dict[str, Any]
        """
    first = frames[0]
    names = {}
    if first._index_width() > 1:
        dtypes = first._dtypes
        for n in first._index_cols:
            names[n] = dtypes[n]
    else:
        mangle = ColNameCodec.mangle_index_names
        idx_names = set()
        for f in frames:
            if f._index_cols is not None:
                idx_names.update(f._index_cols)
            elif f.has_index_cache:
                idx_names.update(mangle(f.index.names))
            else:
                idx_names.add(ColNameCodec.UNNAMED_IDX_COL_NAME)
            if len(idx_names) > 1:
                idx_names = [ColNameCodec.UNNAMED_IDX_COL_NAME]
                break
        name = next(iter(idx_names))
        if first._index_cols is not None:
            names[name] = first._dtypes.iloc[0]
        elif first.has_index_cache:
            names[name] = first.index.dtype
        else:
            names[name] = _get_dtype(int)
    return names