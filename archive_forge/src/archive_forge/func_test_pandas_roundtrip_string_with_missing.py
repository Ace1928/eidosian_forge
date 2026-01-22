from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
import pyarrow.interchange as pi
from pyarrow.interchange.column import (
from pyarrow.interchange.from_dataframe import _from_dataframe
@pytest.mark.pandas
def test_pandas_roundtrip_string_with_missing():
    if Version(pd.__version__) < Version('1.6'):
        pytest.skip('Column.size() bug in pandas')
    arr = ['a', '', 'c', None]
    table = pa.table({'a': pa.array(arr), 'a_large': pa.array(arr, type=pa.large_string())})
    from pandas.api.interchange import from_dataframe as pandas_from_dataframe
    if Version(pd.__version__) >= Version('2.0.2'):
        pandas_df = pandas_from_dataframe(table)
        result = pi.from_dataframe(pandas_df)
        assert result['a'].to_pylist() == table['a'].to_pylist()
        assert pa.types.is_string(table['a'].type)
        assert pa.types.is_large_string(result['a'].type)
        assert result['a_large'].to_pylist() == table['a_large'].to_pylist()
        assert pa.types.is_large_string(table['a_large'].type)
        assert pa.types.is_large_string(result['a_large'].type)
    else:
        with pytest.raises(NotImplementedError):
            pandas_from_dataframe(table)