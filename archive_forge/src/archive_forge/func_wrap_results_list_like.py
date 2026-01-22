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
def wrap_results_list_like(self, keys: Iterable[Hashable], results: list[Series | DataFrame]):
    from pandas.core.reshape.concat import concat
    obj = self.obj
    try:
        return concat(results, keys=keys, axis=1, sort=False)
    except TypeError as err:
        from pandas import Series
        result = Series(results, index=keys, name=obj.name)
        if is_nested_object(result):
            raise ValueError('cannot combine transform and aggregation operations') from err
        return result