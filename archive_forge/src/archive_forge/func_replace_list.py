from __future__ import annotations
from functools import wraps
import inspect
import re
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import missing
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.replace import (
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays import (
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation import expressions
from pandas.core.construction import (
from pandas.core.indexers import check_setitem_lengths
from pandas.core.indexes.base import get_values_for_csv
@final
def replace_list(self, src_list: Iterable[Any], dest_list: Sequence[Any], inplace: bool=False, regex: bool=False, using_cow: bool=False, already_warned=None) -> list[Block]:
    """
        See BlockManager.replace_list docstring.
        """
    values = self.values
    if isinstance(values, Categorical):
        blk = self._maybe_copy(using_cow, inplace)
        values = cast(Categorical, blk.values)
        values._replace(to_replace=src_list, value=dest_list, inplace=True)
        return [blk]
    pairs = [(x, y) for x, y in zip(src_list, dest_list) if self._can_hold_element(x)]
    if not len(pairs):
        if using_cow:
            return [self.copy(deep=False)]
        return [self] if inplace else [self.copy()]
    src_len = len(pairs) - 1
    if is_string_dtype(values.dtype):
        na_mask = ~isna(values)
        masks: Iterable[npt.NDArray[np.bool_]] = (extract_bool_array(cast(ArrayLike, compare_or_regex_search(values, s[0], regex=regex, mask=na_mask))) for s in pairs)
    else:
        masks = (missing.mask_missing(values, s[0]) for s in pairs)
    if inplace:
        masks = list(masks)
    if using_cow:
        rb = [self]
    else:
        rb = [self if inplace else self.copy()]
    if inplace and warn_copy_on_write() and (already_warned is not None) and (not already_warned.warned_already):
        if self.refs.has_reference():
            warnings.warn(COW_WARNING_GENERAL_MSG, FutureWarning, stacklevel=find_stack_level())
            already_warned.warned_already = True
    opt = get_option('future.no_silent_downcasting')
    for i, ((src, dest), mask) in enumerate(zip(pairs, masks)):
        convert = i == src_len
        new_rb: list[Block] = []
        for blk_num, blk in enumerate(rb):
            if len(rb) == 1:
                m = mask
            else:
                mib = mask
                assert not isinstance(mib, bool)
                m = mib[blk_num:blk_num + 1]
            result = blk._replace_coerce(to_replace=src, value=dest, mask=m, inplace=inplace, regex=regex, using_cow=using_cow)
            if using_cow and i != src_len:
                for b in result:
                    ref = weakref.ref(b)
                    b.refs.referenced_blocks.pop(b.refs.referenced_blocks.index(ref))
            if not opt and convert and blk.is_object and (not all((x is None for x in dest_list))):
                nbs = []
                for res_blk in result:
                    converted = res_blk.convert(copy=True and (not using_cow), using_cow=using_cow)
                    if len(converted) > 1 or converted[0].dtype != res_blk.dtype:
                        warnings.warn("Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`", FutureWarning, stacklevel=find_stack_level())
                    nbs.extend(converted)
                result = nbs
            new_rb.extend(result)
        rb = new_rb
    return rb