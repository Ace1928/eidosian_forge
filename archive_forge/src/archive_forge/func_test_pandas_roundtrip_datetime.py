from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
import pyarrow.interchange as pi
from pyarrow.interchange.column import (
from pyarrow.interchange.from_dataframe import _from_dataframe
@pytest.mark.pandas
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_pandas_roundtrip_datetime(unit):
    if Version(pd.__version__) < Version('1.5.0'):
        pytest.skip('__dataframe__ added to pandas in 1.5.0')
    from datetime import datetime as dt
    dt_arr = [dt(2007, 7, 13), dt(2007, 7, 14), dt(2007, 7, 15)]
    table = pa.table({'a': pa.array(dt_arr, type=pa.timestamp(unit))})
    if Version(pd.__version__) < Version('1.6'):
        expected = pa.table({'a': pa.array(dt_arr, type=pa.timestamp('ns'))})
    else:
        expected = table
    from pandas.api.interchange import from_dataframe as pandas_from_dataframe
    pandas_df = pandas_from_dataframe(table)
    result = pi.from_dataframe(pandas_df)
    assert expected.equals(result)
    expected_protocol = expected.__dataframe__()
    result_protocol = result.__dataframe__()
    assert expected_protocol.num_columns() == result_protocol.num_columns()
    assert expected_protocol.num_rows() == result_protocol.num_rows()
    assert expected_protocol.num_chunks() == result_protocol.num_chunks()
    assert expected_protocol.column_names() == result_protocol.column_names()