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
def transform_dict_like(self, func) -> DataFrame:
    """
        Compute transform in the case of a dict-like func
        """
    from pandas.core.reshape.concat import concat
    obj = self.obj
    args = self.args
    kwargs = self.kwargs
    assert isinstance(obj, ABCNDFrame)
    if len(func) == 0:
        raise ValueError('No transform functions were provided')
    func = self.normalize_dictlike_arg('transform', obj, func)
    results: dict[Hashable, DataFrame | Series] = {}
    for name, how in func.items():
        colg = obj._gotitem(name, ndim=1)
        results[name] = colg.transform(how, 0, *args, **kwargs)
    return concat(results, axis=1)