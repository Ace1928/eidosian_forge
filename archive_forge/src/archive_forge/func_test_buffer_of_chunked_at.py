import numpy as np
import pandas
import pyarrow as pa
import pytest
import modin.pandas as pd
from modin.core.dataframe.pandas.interchange.dataframe_protocol.from_dataframe import (
from modin.pandas.io import from_arrow, from_dataframe
from modin.tests.pandas.utils import df_equals
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import export_frame, get_data_of_all_types, split_df_into_chunks
@pytest.mark.parametrize('data_has_nulls', [True, False])
@pytest.mark.parametrize('n_chunks', [2, 9])
def test_buffer_of_chunked_at(data_has_nulls, n_chunks):
    """Test that getting buffers of physically chunked column works properly."""
    data = get_data_of_all_types(has_nulls=data_has_nulls, include_dtypes=['bool', 'int', 'uint', 'float'])
    pd_df = pandas.DataFrame(data)
    pd_chunks = split_df_into_chunks(pd_df, n_chunks)
    chunked_at = pa.concat_tables([pa.Table.from_pandas(pd_df) for pd_df in pd_chunks])
    md_df = from_arrow(chunked_at)
    protocol_df = md_df.__dataframe__()
    for i, col in enumerate(protocol_df.get_columns()):
        assert col.num_chunks() > 1
        assert len(col._pyarrow_table.column(0).chunks) > 1
        buffers = col.get_buffers()
        data_buff, data_dtype = buffers['data']
        result = buffer_to_ndarray(data_buff, data_dtype, col.offset, col.size())
        result = set_nulls(result, col, buffers['validity'])
        with warns_that_defaulting_to_pandas():
            reference = md_df.iloc[:, i].to_numpy()
        np.testing.assert_array_equal(reference, result)
    protocol_df = md_df.__dataframe__(allow_copy=False)
    for i, col in enumerate(protocol_df.get_columns()):
        assert col.num_chunks() > 1
        assert len(col._pyarrow_table.column(0).chunks) > 1
        with pytest.raises(RuntimeError):
            col.get_buffers()