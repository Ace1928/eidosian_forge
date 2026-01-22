from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_little_endian
from pandas import (
import pandas._testing as tm
@pytest.mark.skipif(using_pyarrow_string_dtype(), reason="dtype checking logic doesn't work")
def test_from_records_sequencelike(self):
    df = DataFrame({'A': np.array(np.random.default_rng(2).standard_normal(6), dtype=np.float64), 'A1': np.array(np.random.default_rng(2).standard_normal(6), dtype=np.float64), 'B': np.array(np.arange(6), dtype=np.int64), 'C': ['foo'] * 6, 'D': np.array([True, False] * 3, dtype=bool), 'E': np.array(np.random.default_rng(2).standard_normal(6), dtype=np.float32), 'E1': np.array(np.random.default_rng(2).standard_normal(6), dtype=np.float32), 'F': np.array(np.arange(6), dtype=np.int32)})
    blocks = df._to_dict_of_blocks()
    tuples = []
    columns = []
    dtypes = []
    for dtype, b in blocks.items():
        columns.extend(b.columns)
        dtypes.extend([(c, np.dtype(dtype).descr[0][1]) for c in b.columns])
    for i in range(len(df.index)):
        tup = []
        for _, b in blocks.items():
            tup.extend(b.iloc[i].values)
        tuples.append(tuple(tup))
    recarray = np.array(tuples, dtype=dtypes).view(np.rec.recarray)
    recarray2 = df.to_records()
    lists = [list(x) for x in tuples]
    result = DataFrame.from_records(tuples, columns=columns).reindex(columns=df.columns)
    result2 = DataFrame.from_records(recarray, columns=columns).reindex(columns=df.columns)
    result3 = DataFrame.from_records(recarray2, columns=columns).reindex(columns=df.columns)
    result4 = DataFrame.from_records(lists, columns=columns).reindex(columns=df.columns)
    tm.assert_frame_equal(result, df, check_dtype=False)
    tm.assert_frame_equal(result2, df)
    tm.assert_frame_equal(result3, df)
    tm.assert_frame_equal(result4, df, check_dtype=False)
    result = DataFrame.from_records(tuples)
    tm.assert_index_equal(result.columns, RangeIndex(8))
    columns_to_test = [columns.index('C'), columns.index('E1')]
    exclude = list(set(range(8)) - set(columns_to_test))
    result = DataFrame.from_records(tuples, exclude=exclude)
    result.columns = [columns[i] for i in sorted(columns_to_test)]
    tm.assert_series_equal(result['C'], df['C'])
    tm.assert_series_equal(result['E1'], df['E1'])