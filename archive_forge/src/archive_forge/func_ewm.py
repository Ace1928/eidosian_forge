from __future__ import annotations
import collections
from copy import deepcopy
import datetime as dt
from functools import partial
import gc
from json import loads
import operator
import pickle
import re
import sys
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.lib import is_range_indexer
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.array_algos.replace import should_use_regex
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
@final
@doc(ExponentialMovingWindow)
def ewm(self, com: float | None=None, span: float | None=None, halflife: float | TimedeltaConvertibleTypes | None=None, alpha: float | None=None, min_periods: int | None=0, adjust: bool_t=True, ignore_na: bool_t=False, axis: Axis | lib.NoDefault=lib.no_default, times: np.ndarray | DataFrame | Series | None=None, method: Literal['single', 'table']='single') -> ExponentialMovingWindow:
    if axis is not lib.no_default:
        axis = self._get_axis_number(axis)
        name = 'ewm'
        if axis == 1:
            warnings.warn(f'Support for axis=1 in {type(self).__name__}.{name} is deprecated and will be removed in a future version. Use obj.T.{name}(...) instead', FutureWarning, stacklevel=find_stack_level())
        else:
            warnings.warn(f"The 'axis' keyword in {type(self).__name__}.{name} is deprecated and will be removed in a future version. Call the method without the axis keyword instead.", FutureWarning, stacklevel=find_stack_level())
    else:
        axis = 0
    return ExponentialMovingWindow(self, com=com, span=span, halflife=halflife, alpha=alpha, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na, axis=axis, times=times, method=method)