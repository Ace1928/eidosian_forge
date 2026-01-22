from __future__ import annotations
import abc
from collections import defaultdict
import functools
from functools import partial
import inspect
from typing import (
import warnings
import numpy as np
from pandas._config import option_context
from pandas._libs import lib
from pandas._libs.internals import BlockValuesRefs
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SpecificationError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import is_nested_object
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core._numba.executor import generate_apply_looper
import pandas.core.common as com
from pandas.core.construction import ensure_wrapped_if_datetimelike
def frame_apply(obj: DataFrame, func: AggFuncType, axis: Axis=0, raw: bool=False, result_type: str | None=None, by_row: Literal[False, 'compat']='compat', engine: str='python', engine_kwargs: dict[str, bool] | None=None, args=None, kwargs=None) -> FrameApply:
    """construct and return a row or column based frame apply object"""
    axis = obj._get_axis_number(axis)
    klass: type[FrameApply]
    if axis == 0:
        klass = FrameRowApply
    elif axis == 1:
        klass = FrameColumnApply
    _, func, _, _ = reconstruct_func(func, **kwargs)
    assert func is not None
    return klass(obj, func, raw=raw, result_type=result_type, by_row=by_row, engine=engine, engine_kwargs=engine_kwargs, args=args, kwargs=kwargs)