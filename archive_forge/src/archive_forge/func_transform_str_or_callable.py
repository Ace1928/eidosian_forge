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
def transform_str_or_callable(self, func) -> DataFrame | Series:
    """
        Compute transform in the case of a string or callable func
        """
    obj = self.obj
    args = self.args
    kwargs = self.kwargs
    if isinstance(func, str):
        return self._apply_str(obj, func, *args, **kwargs)
    if not args and (not kwargs):
        f = com.get_cython_func(func)
        if f:
            warn_alias_replacement(obj, func, f)
            return getattr(obj, f)()
    try:
        return obj.apply(func, args=args, **kwargs)
    except Exception:
        return func(obj, *args, **kwargs)