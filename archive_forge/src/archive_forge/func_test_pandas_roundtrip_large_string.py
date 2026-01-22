from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
import pyarrow.interchange as pi
from pyarrow.interchange.column import (
from pyarrow.interchange.from_dataframe import _from_dataframe
@pytest.mark.pandas
def test_pandas_roundtrip_large_string():
    if Version(pd.__version__) < Version('1.6'):
        pytest.skip('Column.size() bug in pandas')
    arr = ['a', '', 'c']
    table = pa.table({'a_large': pa.array(arr, type=pa.large_string())})
    from pandas.api.interchange import from_dataframe as pandas_from_dataframe
    if Version(pd.__version__) >= Version('2.0.1'):
        pandas_df = pandas_from_dataframe(table)
        result = pi.from_dataframe(pandas_df)
        assert result['a_large'].to_pylist() == table['a_large'].to_pylist()
        assert pa.types.is_large_string(table['a_large'].type)
        assert pa.types.is_large_string(result['a_large'].type)
        table_protocol = table.__dataframe__()
        result_protocol = result.__dataframe__()
        assert table_protocol.num_columns() == result_protocol.num_columns()
        assert table_protocol.num_rows() == result_protocol.num_rows()
        assert table_protocol.num_chunks() == result_protocol.num_chunks()
        assert table_protocol.column_names() == result_protocol.column_names()
    else:
        with pytest.raises(AssertionError):
            pandas_from_dataframe(table)