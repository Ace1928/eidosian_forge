from __future__ import annotations
from collections.abc import (
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.tslibs import Timestamp
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import infer_dtype_from_scalar
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import (
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.ops import (
def setitem(self, indexer, value, warn: bool=True) -> Self:
    """
        Set values with indexer.

        For SingleBlockManager, this backs s[indexer] = value
        """
    if isinstance(indexer, np.ndarray) and indexer.ndim > self.ndim:
        raise ValueError(f'Cannot set values with ndim > {self.ndim}')
    if warn and warn_copy_on_write() and (not self._has_no_reference(0)):
        warnings.warn(COW_WARNING_GENERAL_MSG, FutureWarning, stacklevel=find_stack_level())
    elif using_copy_on_write() and (not self._has_no_reference(0)):
        if self.ndim == 2 and isinstance(indexer, tuple):
            blk_loc = self.blklocs[indexer[1]]
            if is_list_like(blk_loc) and blk_loc.ndim == 2:
                blk_loc = np.squeeze(blk_loc, axis=0)
            elif not is_list_like(blk_loc):
                blk_loc = [blk_loc]
            if len(blk_loc) == 0:
                return self.copy(deep=False)
            values = self.blocks[0].values
            if values.ndim == 2:
                values = values[blk_loc]
                self._iset_split_block(0, blk_loc, values)
                self.blocks[0].setitem((indexer[0], np.arange(len(blk_loc))), value)
                return self
        self = self.copy()
    return self.apply('setitem', indexer=indexer, value=value)