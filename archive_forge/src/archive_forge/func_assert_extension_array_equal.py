from __future__ import annotations
import operator
from typing import (
import numpy as np
from pandas._libs import lib
from pandas._libs.missing import is_matching_na
from pandas._libs.sparse import SparseIndex
import pandas._libs.testing as _testing
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
from pandas import (
from pandas.core.arrays import (
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.core.arrays.string_ import StringDtype
from pandas.core.indexes.api import safe_sort_index
from pandas.io.formats.printing import pprint_thing
def assert_extension_array_equal(left, right, check_dtype: bool | Literal['equiv']=True, index_values=None, check_exact: bool | lib.NoDefault=lib.no_default, rtol: float | lib.NoDefault=lib.no_default, atol: float | lib.NoDefault=lib.no_default, obj: str='ExtensionArray') -> None:
    """
    Check that left and right ExtensionArrays are equal.

    Parameters
    ----------
    left, right : ExtensionArray
        The two arrays to compare.
    check_dtype : bool, default True
        Whether to check if the ExtensionArray dtypes are identical.
    index_values : Index | numpy.ndarray, default None
        Optional index (shared by both left and right), used in output.
    check_exact : bool, default False
        Whether to compare number exactly.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'ExtensionArray'
        Specify object name being compared, internally used to show appropriate
        assertion message.

        .. versionadded:: 2.0.0

    Notes
    -----
    Missing values are checked separately from valid values.
    A mask of missing values is computed for each and checked to match.
    The remaining all-valid values are cast to object dtype and checked.

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Series([1, 2, 3, 4])
    >>> b, c = a.array, a.array
    >>> tm.assert_extension_array_equal(b, c)
    """
    if check_exact is lib.no_default and rtol is lib.no_default and (atol is lib.no_default):
        check_exact = is_numeric_dtype(left.dtype) and (not is_float_dtype(left.dtype)) or (is_numeric_dtype(right.dtype) and (not is_float_dtype(right.dtype)))
    elif check_exact is lib.no_default:
        check_exact = False
    rtol = rtol if rtol is not lib.no_default else 1e-05
    atol = atol if atol is not lib.no_default else 1e-08
    assert isinstance(left, ExtensionArray), 'left is not an ExtensionArray'
    assert isinstance(right, ExtensionArray), 'right is not an ExtensionArray'
    if check_dtype:
        assert_attr_equal('dtype', left, right, obj=f'Attributes of {obj}')
    if isinstance(left, DatetimeLikeArrayMixin) and isinstance(right, DatetimeLikeArrayMixin) and (type(right) == type(left)):
        if not check_dtype and left.dtype.kind in 'mM':
            if not isinstance(left.dtype, np.dtype):
                l_unit = cast(DatetimeTZDtype, left.dtype).unit
            else:
                l_unit = np.datetime_data(left.dtype)[0]
            if not isinstance(right.dtype, np.dtype):
                r_unit = cast(DatetimeTZDtype, right.dtype).unit
            else:
                r_unit = np.datetime_data(right.dtype)[0]
            if l_unit != r_unit and compare_mismatched_resolutions(left._ndarray, right._ndarray, operator.eq).all():
                return
        assert_numpy_array_equal(np.asarray(left.asi8), np.asarray(right.asi8), index_values=index_values, obj=obj)
        return
    left_na = np.asarray(left.isna())
    right_na = np.asarray(right.isna())
    assert_numpy_array_equal(left_na, right_na, obj=f'{obj} NA mask', index_values=index_values)
    left_valid = left[~left_na].to_numpy(dtype=object)
    right_valid = right[~right_na].to_numpy(dtype=object)
    if check_exact:
        assert_numpy_array_equal(left_valid, right_valid, obj=obj, index_values=index_values)
    else:
        _testing.assert_almost_equal(left_valid, right_valid, check_dtype=bool(check_dtype), rtol=rtol, atol=atol, obj=obj, index_values=index_values)