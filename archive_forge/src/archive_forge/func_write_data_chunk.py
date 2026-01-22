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
def write_data_chunk(self, rows: np.ndarray, indexes: list[np.ndarray], mask: npt.NDArray[np.bool_] | None, values: list[np.ndarray]) -> None:
    """
        Parameters
        ----------
        rows : an empty memory space where we are putting the chunk
        indexes : an array of the indexes
        mask : an array of the masks
        values : an array of the values
        """
    for v in values:
        if not np.prod(v.shape):
            return
    nrows = indexes[0].shape[0]
    if nrows != len(rows):
        rows = np.empty(nrows, dtype=self.dtype)
    names = self.dtype.names
    nindexes = len(indexes)
    for i, idx in enumerate(indexes):
        rows[names[i]] = idx
    for i, v in enumerate(values):
        rows[names[i + nindexes]] = v
    if mask is not None:
        m = ~mask.ravel().astype(bool, copy=False)
        if not m.all():
            rows = rows[m]
    if len(rows):
        self.table.append(rows)
        self.table.flush()