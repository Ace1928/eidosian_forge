import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_arrow_table_roundtrip():
    from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
    arr = PeriodArray([1, 2, 3], dtype='period[D]')
    arr[1] = pd.NaT
    df = pd.DataFrame({'a': arr})
    table = pa.table(df)
    assert isinstance(table.field('a').type, ArrowPeriodType)
    result = table.to_pandas()
    assert isinstance(result['a'].dtype, PeriodDtype)
    tm.assert_frame_equal(result, df)
    table2 = pa.concat_tables([table, table])
    result = table2.to_pandas()
    expected = pd.concat([df, df], ignore_index=True)
    tm.assert_frame_equal(result, expected)