import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.indexes.accessors import Properties
@pytest.mark.parametrize('idx', [date_range('1/1/2015', periods=5), date_range('1/1/2015', periods=5, tz='MET'), period_range('1/1/2015', freq='D', periods=5), timedelta_range('1 days', '10 days')])
def test_dt_accessor_api_for_categorical(self, idx):
    ser = Series(idx)
    cat = ser.astype('category')
    attr_names = type(ser._values)._datetimelike_ops
    assert isinstance(cat.dt, Properties)
    special_func_defs = [('strftime', ('%Y-%m-%d',), {}), ('round', ('D',), {}), ('floor', ('D',), {}), ('ceil', ('D',), {}), ('asfreq', ('D',), {}), ('as_unit', 's', {})]
    if idx.dtype == 'M8[ns]':
        tup = ('tz_localize', ('UTC',), {})
        special_func_defs.append(tup)
    elif idx.dtype.kind == 'M':
        tup = ('tz_convert', ('EST',), {})
        special_func_defs.append(tup)
    _special_func_names = [f[0] for f in special_func_defs]
    _ignore_names = ['components', 'tz_localize', 'tz_convert']
    func_names = [fname for fname in dir(ser.dt) if not (fname.startswith('_') or fname in attr_names or fname in _special_func_names or (fname in _ignore_names))]
    func_defs = [(fname, (), {}) for fname in func_names]
    func_defs.extend((f_def for f_def in special_func_defs if f_def[0] in dir(ser.dt)))
    for func, args, kwargs in func_defs:
        warn_cls = []
        if func == 'to_period' and getattr(idx, 'tz', None) is not None:
            warn_cls.append(UserWarning)
        if func == 'to_pydatetime':
            warn_cls.append(FutureWarning)
        if warn_cls:
            warn_cls = tuple(warn_cls)
        else:
            warn_cls = None
        with tm.assert_produces_warning(warn_cls):
            res = getattr(cat.dt, func)(*args, **kwargs)
            exp = getattr(ser.dt, func)(*args, **kwargs)
        tm.assert_equal(res, exp)
    for attr in attr_names:
        res = getattr(cat.dt, attr)
        exp = getattr(ser.dt, attr)
        tm.assert_equal(res, exp)