import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def is_nullable_dtype(dtype: PandasDType) -> bool:
    """Whether dtype is a pandas nullable type."""
    from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype
    is_int = is_integer_dtype(dtype) and dtype.name in pandas_nullable_mapper
    is_bool = is_bool_dtype(dtype) and dtype.name == 'boolean'
    is_float = is_float_dtype(dtype) and dtype.name in pandas_nullable_mapper
    return is_int or is_bool or is_float or is_pd_cat_dtype(dtype)