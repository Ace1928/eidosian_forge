import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('columns', (['first_name', 'middle_name'], [0, 1]))
def test_data_frame_value_counts_subset(nulls_fixture, columns):
    df = pd.DataFrame({columns[0]: ['John', 'Anne', 'John', 'Beth'], columns[1]: ['Smith', nulls_fixture, nulls_fixture, 'Louise']})
    result = df.value_counts(columns[0])
    expected = pd.Series(data=[2, 1, 1], index=pd.Index(['John', 'Anne', 'Beth'], name=columns[0]), name='count')
    tm.assert_series_equal(result, expected)