import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('copy', [True, None, False])
@pytest.mark.parametrize('method', [lambda ser, copy: ser.rename(index={0: 100}, copy=copy), lambda ser, copy: ser.rename(None, copy=copy), lambda ser, copy: ser.reindex(index=ser.index, copy=copy), lambda ser, copy: ser.reindex_like(ser, copy=copy), lambda ser, copy: ser.align(ser, copy=copy)[0], lambda ser, copy: ser.set_axis(['a', 'b', 'c'], axis='index', copy=copy), lambda ser, copy: ser.rename_axis(index='test', copy=copy), lambda ser, copy: ser.astype('int64', copy=copy), lambda ser, copy: ser.swaplevel(0, 1, copy=copy), lambda ser, copy: ser.swapaxes(0, 0, copy=copy), lambda ser, copy: ser.truncate(0, 5, copy=copy), lambda ser, copy: ser.infer_objects(copy=copy), lambda ser, copy: ser.to_timestamp(copy=copy), lambda ser, copy: ser.to_period(freq='D', copy=copy), lambda ser, copy: ser.tz_localize('US/Central', copy=copy), lambda ser, copy: ser.tz_convert('US/Central', copy=copy), lambda ser, copy: ser.set_flags(allows_duplicate_labels=False, copy=copy)], ids=['rename (dict)', 'rename', 'reindex', 'reindex_like', 'align', 'set_axis', 'rename_axis0', 'astype', 'swaplevel', 'swapaxes', 'truncate', 'infer_objects', 'to_timestamp', 'to_period', 'tz_localize', 'tz_convert', 'set_flags'])
def test_methods_series_copy_keyword(request, method, copy, using_copy_on_write):
    index = None
    if 'to_timestamp' in request.node.callspec.id:
        index = period_range('2012-01-01', freq='D', periods=3)
    elif 'to_period' in request.node.callspec.id:
        index = date_range('2012-01-01', freq='D', periods=3)
    elif 'tz_localize' in request.node.callspec.id:
        index = date_range('2012-01-01', freq='D', periods=3)
    elif 'tz_convert' in request.node.callspec.id:
        index = date_range('2012-01-01', freq='D', periods=3, tz='Europe/Brussels')
    elif 'swaplevel' in request.node.callspec.id:
        index = MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]])
    ser = Series([1, 2, 3], index=index)
    if 'swapaxes' in request.node.callspec.id:
        msg = "'Series.swapaxes' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            ser2 = method(ser, copy=copy)
    else:
        ser2 = method(ser, copy=copy)
    share_memory = using_copy_on_write or copy is False
    if share_memory:
        assert np.shares_memory(get_array(ser2), get_array(ser))
    else:
        assert not np.shares_memory(get_array(ser2), get_array(ser))