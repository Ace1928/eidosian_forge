from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
@pytest.mark.parametrize('columns', ['C', ['C']])
@pytest.mark.parametrize('keys', [['A'], ['A', 'B']])
@pytest.mark.parametrize('values', [[True], [0], [0.0], ['a'], Categorical([0]), [to_datetime(0)], date_range(0, 1, 1, tz='US/Eastern'), pd.period_range('2016-01-01', periods=3, freq='D'), pd.array([0], dtype='Int64'), pd.array([0], dtype='Float64'), pd.array([False], dtype='boolean')], ids=['bool', 'int', 'float', 'str', 'cat', 'dt64', 'dt64tz', 'period', 'Int64', 'Float64', 'boolean'])
@pytest.mark.parametrize('method', ['attr', 'agg', 'apply'])
@pytest.mark.parametrize('op', ['idxmax', 'idxmin', 'min', 'max', 'sum', 'prod', 'skew'])
def test_empty_groupby(columns, keys, values, method, op, using_array_manager, dropna, using_infer_string):
    override_dtype = None
    if isinstance(values, BooleanArray) and op in ['sum', 'prod']:
        override_dtype = 'Int64'
    if isinstance(values[0], bool) and op in ('prod', 'sum'):
        override_dtype = 'int64'
    df = DataFrame({'A': values, 'B': values, 'C': values}, columns=list('ABC'))
    if hasattr(values, 'dtype'):
        assert (df.dtypes == values.dtype).all()
    df = df.iloc[:0]
    gb = df.groupby(keys, group_keys=False, dropna=dropna, observed=False)[columns]

    def get_result(**kwargs):
        if method == 'attr':
            return getattr(gb, op)(**kwargs)
        else:
            return getattr(gb, method)(op, **kwargs)

    def get_categorical_invalid_expected():
        lev = Categorical([0], dtype=values.dtype)
        if len(keys) != 1:
            idx = MultiIndex.from_product([lev, lev], names=keys)
        else:
            idx = Index(lev, name=keys[0])
        if using_infer_string:
            columns = Index([], dtype='string[pyarrow_numpy]')
        else:
            columns = []
        expected = DataFrame([], columns=columns, index=idx)
        return expected
    is_per = isinstance(df.dtypes.iloc[0], pd.PeriodDtype)
    is_dt64 = df.dtypes.iloc[0].kind == 'M'
    is_cat = isinstance(values, Categorical)
    if isinstance(values, Categorical) and (not values.ordered) and (op in ['min', 'max', 'idxmin', 'idxmax']):
        if op in ['min', 'max']:
            msg = f'Cannot perform {op} with non-ordered Categorical'
            klass = TypeError
        else:
            msg = f"Can't get {op} of an empty group due to unobserved categories"
            klass = ValueError
        with pytest.raises(klass, match=msg):
            get_result()
        if op in ['min', 'max', 'idxmin', 'idxmax'] and isinstance(columns, list):
            result = get_result(numeric_only=True)
            expected = get_categorical_invalid_expected()
            tm.assert_equal(result, expected)
        return
    if op in ['prod', 'sum', 'skew']:
        if is_dt64 or is_cat or is_per:
            if is_dt64:
                msg = 'datetime64 type does not support'
            elif is_per:
                msg = 'Period type does not support'
            else:
                msg = 'category type does not support'
            if op == 'skew':
                msg = '|'.join([msg, "does not support reduction 'skew'"])
            with pytest.raises(TypeError, match=msg):
                get_result()
            if not isinstance(columns, list):
                return
            elif op == 'skew':
                return
            else:
                result = get_result(numeric_only=True)
                expected = df.set_index(keys)[[]]
                if is_cat:
                    expected = get_categorical_invalid_expected()
                tm.assert_equal(result, expected)
                return
    result = get_result()
    expected = df.set_index(keys)[columns]
    if op in ['idxmax', 'idxmin']:
        expected = expected.astype(df.index.dtype)
    if override_dtype is not None:
        expected = expected.astype(override_dtype)
    if len(keys) == 1:
        expected.index.name = keys[0]
    tm.assert_equal(result, expected)