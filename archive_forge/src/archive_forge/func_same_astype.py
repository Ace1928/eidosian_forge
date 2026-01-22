from __future__ import annotations
import re
import string
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, cast
import numpy as np
import pandas as pd
from dask.dataframe._compat import PANDAS_GE_220, PANDAS_GE_300
from dask.dataframe._pyarrow import is_object_string_dtype
from dask.dataframe.core import tokenize
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.utils import random_state_data
def same_astype(a: str | type, b: str | type):
    """Same as pandas.api.types.is_dtype_equal, but also returns True for str / object"""
    return pd.api.types.is_dtype_equal(a, b) or (is_object_string_dtype(a) and is_object_string_dtype(b))