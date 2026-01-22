import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('sequence_index', range(3 ** 4))
@pytest.mark.parametrize('dtype', [None, 'UInt8', 'Int8', 'UInt16', 'Int16', 'UInt32', 'Int32', 'UInt64', 'Int64', 'Float32', 'Int64', 'Float64', 'category', 'string', pytest.param('string[pyarrow]', marks=pytest.mark.skipif(pa_version_under10p1, reason='pyarrow is not installed')), 'datetime64[ns]', 'period[d]', 'Sparse[float]'])
@pytest.mark.parametrize('test_series', [True, False])
def test_no_sort_keep_na(sequence_index, dtype, test_series, as_index):
    sequence = ''.join([{0: 'x', 1: 'y', 2: 'z'}[sequence_index // 3 ** k % 3] for k in range(4)])
    if dtype in ('string', 'string[pyarrow]'):
        uniques = {'x': 'x', 'y': 'y', 'z': pd.NA}
    elif dtype in ('datetime64[ns]', 'period[d]'):
        uniques = {'x': '2016-01-01', 'y': '2017-01-01', 'z': pd.NA}
    else:
        uniques = {'x': 1, 'y': 2, 'z': np.nan}
    df = pd.DataFrame({'key': pd.Series([uniques[label] for label in sequence], dtype=dtype), 'a': [0, 1, 2, 3]})
    gb = df.groupby('key', dropna=False, sort=False, as_index=as_index, observed=False)
    if test_series:
        gb = gb['a']
    result = gb.sum()
    summed = {}
    for idx, label in enumerate(sequence):
        summed[label] = summed.get(label, 0) + idx
    if dtype == 'category':
        index = pd.CategoricalIndex([uniques[e] for e in summed], df['key'].cat.categories, name='key')
    elif isinstance(dtype, str) and dtype.startswith('Sparse'):
        index = pd.Index(pd.array([uniques[label] for label in summed], dtype=dtype), name='key')
    else:
        index = pd.Index([uniques[label] for label in summed], dtype=dtype, name='key')
    expected = pd.Series(summed.values(), index=index, name='a', dtype=None)
    if not test_series:
        expected = expected.to_frame()
    if not as_index:
        expected = expected.reset_index()
        if dtype is not None and dtype.startswith('Sparse'):
            expected['key'] = expected['key'].astype(dtype)
    tm.assert_equal(result, expected)