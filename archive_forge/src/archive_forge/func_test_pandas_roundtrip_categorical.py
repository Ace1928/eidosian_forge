from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
import pyarrow.interchange as pi
from pyarrow.interchange.column import (
from pyarrow.interchange.from_dataframe import _from_dataframe
@pytest.mark.pandas
def test_pandas_roundtrip_categorical():
    if Version(pd.__version__) < Version('2.0.2'):
        pytest.skip('Bitmasks not supported in pandas interchange implementation')
    arr = ['Mon', 'Tue', 'Mon', 'Wed', 'Mon', 'Thu', 'Fri', 'Sat', None]
    table = pa.table({'weekday': pa.array(arr).dictionary_encode()})
    from pandas.api.interchange import from_dataframe as pandas_from_dataframe
    pandas_df = pandas_from_dataframe(table)
    result = pi.from_dataframe(pandas_df)
    assert result['weekday'].to_pylist() == table['weekday'].to_pylist()
    assert pa.types.is_dictionary(table['weekday'].type)
    assert pa.types.is_dictionary(result['weekday'].type)
    assert pa.types.is_string(table['weekday'].chunk(0).dictionary.type)
    assert pa.types.is_large_string(result['weekday'].chunk(0).dictionary.type)
    assert pa.types.is_int32(table['weekday'].chunk(0).indices.type)
    assert pa.types.is_int8(result['weekday'].chunk(0).indices.type)
    table_protocol = table.__dataframe__()
    result_protocol = result.__dataframe__()
    assert table_protocol.num_columns() == result_protocol.num_columns()
    assert table_protocol.num_rows() == result_protocol.num_rows()
    assert table_protocol.num_chunks() == result_protocol.num_chunks()
    assert table_protocol.column_names() == result_protocol.column_names()
    col_table = table_protocol.get_column(0)
    col_result = result_protocol.get_column(0)
    assert col_result.dtype[0] == DtypeKind.CATEGORICAL
    assert col_result.dtype[0] == col_table.dtype[0]
    assert col_result.size() == col_table.size()
    assert col_result.offset == col_table.offset
    desc_cat_table = col_result.describe_categorical
    desc_cat_result = col_result.describe_categorical
    assert desc_cat_table['is_ordered'] == desc_cat_result['is_ordered']
    assert desc_cat_table['is_dictionary'] == desc_cat_result['is_dictionary']
    assert isinstance(desc_cat_result['categories']._col, pa.Array)