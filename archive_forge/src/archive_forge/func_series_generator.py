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
@property
def series_generator(self) -> Generator[Series, None, None]:
    values = self.values
    values = ensure_wrapped_if_datetimelike(values)
    assert len(values) > 0
    ser = self.obj._ixs(0, axis=0)
    mgr = ser._mgr
    is_view = mgr.blocks[0].refs.has_reference()
    if isinstance(ser.dtype, ExtensionDtype):
        obj = self.obj
        for i in range(len(obj)):
            yield obj._ixs(i, axis=0)
    else:
        for arr, name in zip(values, self.index):
            ser._mgr = mgr
            mgr.set_values(arr)
            object.__setattr__(ser, '_name', name)
            if not is_view:
                mgr.blocks[0].refs = BlockValuesRefs(mgr.blocks[0])
            yield ser