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
def normalize_dictlike_arg(self, how: str, obj: DataFrame | Series, func: AggFuncTypeDict) -> AggFuncTypeDict:
    """
        Handler for dict-like argument.

        Ensures that necessary columns exist if obj is a DataFrame, and
        that a nested renamer is not passed. Also normalizes to all lists
        when values consists of a mix of list and non-lists.
        """
    assert how in ('apply', 'agg', 'transform')
    if how == 'agg' and isinstance(obj, ABCSeries) and any((is_list_like(v) for _, v in func.items())) or any((is_dict_like(v) for _, v in func.items())):
        raise SpecificationError('nested renamer is not supported')
    if obj.ndim != 1:
        from pandas import Index
        cols = Index(list(func.keys())).difference(obj.columns, sort=True)
        if len(cols) > 0:
            raise KeyError(f'Column(s) {list(cols)} do not exist')
    aggregator_types = (list, tuple, dict)
    if any((isinstance(x, aggregator_types) for _, x in func.items())):
        new_func: AggFuncTypeDict = {}
        for k, v in func.items():
            if not isinstance(v, aggregator_types):
                new_func[k] = [v]
            else:
                new_func[k] = v
        func = new_func
    return func