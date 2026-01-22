from __future__ import annotations
from collections import abc
from typing import (
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core import (
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
from pandas.core.indexes.api import (
from pandas.core.internals.array_manager import (
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def dict_to_mgr(data: dict, index, columns, *, dtype: DtypeObj | None=None, typ: str='block', copy: bool=True) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.
    Needs to handle a lot of exceptional cases.

    Used in DataFrame.__init__
    """
    arrays: Sequence[Any] | Series
    if columns is not None:
        from pandas.core.series import Series
        arrays = Series(data, index=columns, dtype=object)
        missing = arrays.isna()
        if index is None:
            index = _extract_index(arrays[~missing])
        else:
            index = ensure_index(index)
        if missing.any() and (not is_integer_dtype(dtype)):
            nan_dtype: DtypeObj
            if dtype is not None:
                midxs = missing.values.nonzero()[0]
                for i in midxs:
                    arr = sanitize_array(arrays.iat[i], index, dtype=dtype)
                    arrays.iat[i] = arr
            else:
                nan_dtype = np.dtype('object')
                val = construct_1d_arraylike_from_scalar(np.nan, len(index), nan_dtype)
                nmissing = missing.sum()
                if copy:
                    rhs = [val] * nmissing
                else:
                    rhs = [val.copy() for _ in range(nmissing)]
                arrays.loc[missing] = rhs
        arrays = list(arrays)
        columns = ensure_index(columns)
    else:
        keys = list(data.keys())
        columns = Index(keys) if keys else default_index(0)
        arrays = [com.maybe_iterable_to_list(data[k]) for k in keys]
    if copy:
        if typ == 'block':
            arrays = [x.copy() if isinstance(x, ExtensionArray) else x.copy(deep=True) if isinstance(x, Index) or (isinstance(x, ABCSeries) and is_1d_only_ea_dtype(x.dtype)) else x for x in arrays]
        else:
            arrays = [x.copy() if hasattr(x, 'dtype') else x for x in arrays]
    return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)