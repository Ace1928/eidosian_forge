from __future__ import annotations
import contextlib
import functools
import math
from collections import defaultdict
from collections.abc import Collection, Hashable, Sequence
from datetime import datetime, timedelta
from itertools import chain, zip_longest
from reprlib import recursive_repr
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from xarray.core.duck_array_ops import array_equiv, astype
from xarray.core.indexing import MemoryCachedArray
from xarray.core.options import OPTIONS, _get_boolean_with_default
from xarray.core.utils import is_duck_array
from xarray.namedarray.pycompat import array_type, to_duck_array, to_numpy
def short_array_repr(array):
    from xarray.core.common import AbstractArray
    if isinstance(array, AbstractArray):
        array = array.data
    array = to_duck_array(array)
    options = {'precision': 6, 'linewidth': OPTIONS['display_width'], 'threshold': OPTIONS['display_values_threshold']}
    if array.ndim < 3:
        edgeitems = 3
    elif array.ndim == 3:
        edgeitems = 2
    else:
        edgeitems = 1
    options['edgeitems'] = edgeitems
    with set_numpy_options(**options):
        return repr(array)