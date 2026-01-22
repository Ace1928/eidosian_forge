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
def remove_unused_levels(self) -> MultiIndex:
    """
        Create new MultiIndex from current that removes unused levels.

        Unused level(s) means levels that are not expressed in the
        labels. The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will
        also be .equals() to the original.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex.from_product([range(2), list('ab')])
        >>> mi
        MultiIndex([(0, 'a'),
                    (0, 'b'),
                    (1, 'a'),
                    (1, 'b')],
                   )

        >>> mi[2:]
        MultiIndex([(1, 'a'),
                    (1, 'b')],
                   )

        The 0 from the first level is not represented
        and can be removed

        >>> mi2 = mi[2:].remove_unused_levels()
        >>> mi2.levels
        FrozenList([[1], ['a', 'b']])
        """
    new_levels = []
    new_codes = []
    changed = False
    for lev, level_codes in zip(self.levels, self.codes):
        uniques = np.where(np.bincount(level_codes + 1) > 0)[0] - 1
        has_na = int(len(uniques) and uniques[0] == -1)
        if len(uniques) != len(lev) + has_na:
            if lev.isna().any() and len(uniques) == len(lev):
                break
            changed = True
            uniques = algos.unique(level_codes)
            if has_na:
                na_idx = np.where(uniques == -1)[0]
                uniques[[0, na_idx[0]]] = uniques[[na_idx[0], 0]]
            code_mapping = np.zeros(len(lev) + has_na)
            code_mapping[uniques] = np.arange(len(uniques)) - has_na
            level_codes = code_mapping[level_codes]
            lev = lev.take(uniques[has_na:])
        new_levels.append(lev)
        new_codes.append(level_codes)
    result = self.view()
    if changed:
        result._reset_identity()
        result._set_levels(new_levels, validate=False)
        result._set_codes(new_codes, validate=False)
    return result