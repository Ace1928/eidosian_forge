from __future__ import annotations
from collections.abc import (
import datetime
from functools import (
import inspect
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config.config import option_context
from pandas._libs import (
from pandas._libs.algos import rank_1d
import pandas._libs.groupby as libgroupby
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core._numba import executor
from pandas.core.apply import warn_alias_replacement
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
from pandas.core.arrays.string_arrow import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import (
from pandas.core.groupby.grouper import get_grouper
from pandas.core.groupby.indexing import (
from pandas.core.indexes.api import (
from pandas.core.internals.blocks import ensure_block_shape
from pandas.core.series import Series
from pandas.core.sorting import get_group_index_sorter
from pandas.core.util.numba_ import (
def hfunc(bvalues: ArrayLike) -> ArrayLike:
    if bvalues.ndim == 1:
        masked = mask & ~isna(bvalues).reshape(1, -1)
    else:
        masked = mask & ~isna(bvalues)
    counted = lib.count_level_2d(masked, labels=ids, max_bin=ngroups)
    if isinstance(bvalues, BaseMaskedArray):
        return IntegerArray(counted[0], mask=np.zeros(counted.shape[1], dtype=np.bool_))
    elif isinstance(bvalues, ArrowExtensionArray) and (not isinstance(bvalues.dtype, StringDtype)):
        dtype = pandas_dtype('int64[pyarrow]')
        return type(bvalues)._from_sequence(counted[0], dtype=dtype)
    if is_series:
        assert counted.ndim == 2
        assert counted.shape[0] == 1
        return counted[0]
    return counted