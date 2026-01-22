import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
@pytest.mark.parametrize('indexer,expected_slice', [(lambda df: df['foo'], slice(3)), (lambda df: df['bar'], slice(3, 5)), (lambda df: df.loc[:, 'bar'], slice(3, 5))])
def test_frame_getitem_toplevel(multiindex_dataframe_random_data, indexer, expected_slice):
    df = multiindex_dataframe_random_data.T
    expected = df.reindex(columns=df.columns[expected_slice])
    expected.columns = expected.columns.droplevel(0)
    result = indexer(df)
    tm.assert_frame_equal(result, expected)