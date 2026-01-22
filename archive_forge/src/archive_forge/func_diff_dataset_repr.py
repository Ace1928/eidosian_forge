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
def diff_dataset_repr(a, b, compat):
    summary = [f'Left and right {type(a).__name__} objects are not {_compat_to_str(compat)}']
    col_width = _calculate_col_width(set(list(a.variables) + list(b.variables)))
    summary.append(diff_dim_summary(a, b))
    summary.append(diff_coords_repr(a.coords, b.coords, compat, col_width=col_width))
    summary.append(diff_data_vars_repr(a.data_vars, b.data_vars, compat, col_width=col_width))
    if compat == 'identical':
        summary.append(diff_attrs_repr(a.attrs, b.attrs, compat))
    return '\n'.join(summary)