from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('keys', ['A', 'C', ['A', 'B'], ('tuple', 'as', 'label')])
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('drop', [True, False])
def test_set_index_drop_inplace(self, frame_of_index_cols, drop, inplace, keys):
    df = frame_of_index_cols
    if isinstance(keys, list):
        idx = MultiIndex.from_arrays([df[x] for x in keys], names=keys)
    else:
        idx = Index(df[keys], name=keys)
    expected = df.drop(keys, axis=1) if drop else df
    expected.index = idx
    if inplace:
        result = df.copy()
        return_value = result.set_index(keys, drop=drop, inplace=True)
        assert return_value is None
    else:
        result = df.set_index(keys, drop=drop)
    tm.assert_frame_equal(result, expected)