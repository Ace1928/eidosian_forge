import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_data_frame_value_counts_dropna_true(nulls_fixture):
    df = pd.DataFrame({'first_name': ['John', 'Anne', 'John', 'Beth'], 'middle_name': ['Smith', nulls_fixture, nulls_fixture, 'Louise']})
    result = df.value_counts()
    expected = pd.Series(data=[1, 1], index=pd.MultiIndex.from_arrays([('Beth', 'John'), ('Louise', 'Smith')], names=['first_name', 'middle_name']), name='count')
    tm.assert_series_equal(result, expected)