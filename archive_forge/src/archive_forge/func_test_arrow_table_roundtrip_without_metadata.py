import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_arrow_table_roundtrip_without_metadata():
    arr = PeriodArray([1, 2, 3], dtype='period[h]')
    arr[1] = pd.NaT
    df = pd.DataFrame({'a': arr})
    table = pa.table(df)
    table = table.replace_schema_metadata()
    assert table.schema.metadata is None
    result = table.to_pandas()
    assert isinstance(result['a'].dtype, PeriodDtype)
    tm.assert_frame_equal(result, df)