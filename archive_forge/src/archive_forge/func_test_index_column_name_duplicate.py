import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_index_column_name_duplicate(tempdir):
    data = {'close': {pd.Timestamp('2017-06-30 01:31:00'): 154.99958999999998, pd.Timestamp('2017-06-30 01:32:00'): 154.99958999999998}, 'time': {pd.Timestamp('2017-06-30 01:31:00'): pd.Timestamp('2017-06-30 01:31:00'), pd.Timestamp('2017-06-30 01:32:00'): pd.Timestamp('2017-06-30 01:32:00')}}
    path = str(tempdir / 'data.parquet')
    dfx = pd.DataFrame(data, dtype='datetime64[us]').set_index('time', drop=False)
    tdfx = pa.Table.from_pandas(dfx)
    _write_table(tdfx, path)
    arrow_table = _read_table(path)
    result_df = arrow_table.to_pandas()
    tm.assert_frame_equal(result_df, dfx)