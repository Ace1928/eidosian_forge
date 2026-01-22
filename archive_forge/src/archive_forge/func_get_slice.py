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
def get_slice(self, slobj: slice, axis: AxisInt=0) -> SingleBlockManager:
    if axis >= self.ndim:
        raise IndexError('Requested axis not found in manager')
    blk = self._block
    array = blk.values[slobj]
    bp = BlockPlacement(slice(0, len(array)))
    block = type(blk)(array, placement=bp, ndim=1, refs=blk.refs)
    new_index = self.index._getitem_slice(slobj)
    return type(self)(block, new_index)