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
def wrap_results_dict_like(self, selected_obj: Series | DataFrame, result_index: list[Hashable], result_data: list):
    from pandas import Index
    from pandas.core.reshape.concat import concat
    obj = self.obj
    is_ndframe = [isinstance(r, ABCNDFrame) for r in result_data]
    if all(is_ndframe):
        results = dict(zip(result_index, result_data))
        keys_to_use: Iterable[Hashable]
        keys_to_use = [k for k in result_index if not results[k].empty]
        keys_to_use = keys_to_use if keys_to_use != [] else result_index
        if selected_obj.ndim == 2:
            ktu = Index(keys_to_use)
            ktu._set_names(selected_obj.columns.names)
            keys_to_use = ktu
        axis: AxisInt = 0 if isinstance(obj, ABCSeries) else 1
        result = concat({k: results[k] for k in keys_to_use}, axis=axis, keys=keys_to_use)
    elif any(is_ndframe):
        raise ValueError('cannot perform both aggregation and transformation operations simultaneously')
    else:
        from pandas import Series
        if obj.ndim == 1:
            obj = cast('Series', obj)
            name = obj.name
        else:
            name = None
        result = Series(result_data, index=result_index, name=name)
    return result