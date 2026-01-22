from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
def read_array(self, key: str, start: int | None=None, stop: int | None=None):
    """read an array for the specified node (off of group"""
    import tables
    node = getattr(self.group, key)
    attrs = node._v_attrs
    transposed = getattr(attrs, 'transposed', False)
    if isinstance(node, tables.VLArray):
        ret = node[0][start:stop]
    else:
        dtype = _ensure_decoded(getattr(attrs, 'value_type', None))
        shape = getattr(attrs, 'shape', None)
        if shape is not None:
            ret = np.empty(shape, dtype=dtype)
        else:
            ret = node[start:stop]
        if dtype and dtype.startswith('datetime64'):
            tz = getattr(attrs, 'tz', None)
            ret = _set_tz(ret, tz, coerce=True)
        elif dtype == 'timedelta64':
            ret = np.asarray(ret, dtype='m8[ns]')
    if transposed:
        return ret.T
    else:
        return ret