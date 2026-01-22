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
def build_categorical_from_at(table, column_name):
    """
    Build ``pandas.CategoricalDtype`` from a dictionary column of the passed PyArrow Table.

    Parameters
    ----------
    table : pyarrow.Table
    column_name : str

    Returns
    -------
    pandas.CategoricalDtype
    """
    chunks = table.column(column_name).chunks
    cat = pandas.concat([chunk.dictionary.to_pandas() for chunk in chunks])
    del chunks
    return pandas.CategoricalDtype(cat.unique())