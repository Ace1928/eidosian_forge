from __future__ import annotations
import itertools
from typing import (
import warnings
import numpy as np
import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.algorithms import (
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
def get_new_columns(self, value_columns: Index | None):
    if value_columns is None:
        if self.lift == 0:
            return self.removed_level._rename(name=self.removed_name)
        lev = self.removed_level.insert(0, item=self.removed_level._na_value)
        return lev.rename(self.removed_name)
    stride = len(self.removed_level) + self.lift
    width = len(value_columns)
    propagator = np.repeat(np.arange(width), stride)
    new_levels: FrozenList | list[Index]
    if isinstance(value_columns, MultiIndex):
        new_levels = value_columns.levels + (self.removed_level_full,)
        new_names = value_columns.names + (self.removed_name,)
        new_codes = [lab.take(propagator) for lab in value_columns.codes]
    else:
        new_levels = [value_columns, self.removed_level_full]
        new_names = [value_columns.name, self.removed_name]
        new_codes = [propagator]
    repeater = self._repeater
    new_codes.append(np.tile(repeater, width))
    return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)