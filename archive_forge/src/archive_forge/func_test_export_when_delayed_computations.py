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
def test_export_when_delayed_computations():
    """
    Test that export works properly when HdkOnNative has delayed computations.

    If there are delayed functions and export is required, it has to trigger the execution
    first prior materializing protocol's buffers, so the buffers contain actual result
    of the computations.
    """
    data = get_data_of_all_types(has_nulls=True, exclude_dtypes=['uint64', 'bool'])
    md_df = pd.DataFrame(data)
    pd_df = pandas.DataFrame(data)
    md_res = md_df.fillna({'float32_null': 32.0, 'float64_null': 64.0})
    pd_res = pd_df.fillna({'float32_null': 32.0, 'float64_null': 64.0})
    assert not md_res._query_compiler._modin_frame._has_arrow_table(), 'There are no delayed computations for the frame'
    exported_df = export_frame(md_res)
    df_equals(exported_df, pd_res)