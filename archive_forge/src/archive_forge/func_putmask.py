from __future__ import annotations
from collections.abc import (
from functools import wraps
from sys import getsizeof
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import coerce_indexer_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import validate_putmask
from pandas.core.arrays import (
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.construction import sanitize_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.io.formats.printing import (
def putmask(self, mask, value: MultiIndex) -> MultiIndex:
    """
        Return a new MultiIndex of the values set with the mask.

        Parameters
        ----------
        mask : array like
        value : MultiIndex
            Must either be the same length as self or length one

        Returns
        -------
        MultiIndex
        """
    mask, noop = validate_putmask(self, mask)
    if noop:
        return self.copy()
    if len(mask) == len(value):
        subset = value[mask].remove_unused_levels()
    else:
        subset = value.remove_unused_levels()
    new_levels = []
    new_codes = []
    for i, (value_level, level, level_codes) in enumerate(zip(subset.levels, self.levels, self.codes)):
        new_level = level.union(value_level, sort=False)
        value_codes = new_level.get_indexer_for(subset.get_level_values(i))
        new_code = ensure_int64(level_codes)
        new_code[mask] = value_codes
        new_levels.append(new_level)
        new_codes.append(new_code)
    return MultiIndex(levels=new_levels, codes=new_codes, names=self.names, verify_integrity=False)