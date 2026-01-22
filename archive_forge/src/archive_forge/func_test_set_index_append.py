from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('keys', ['A', 'C', ['A', 'B'], ('tuple', 'as', 'label')])
@pytest.mark.parametrize('drop', [True, False])
def test_set_index_append(self, frame_of_index_cols, drop, keys):
    df = frame_of_index_cols
    keys = keys if isinstance(keys, list) else [keys]
    idx = MultiIndex.from_arrays([df.index] + [df[x] for x in keys], names=[None] + keys)
    expected = df.drop(keys, axis=1) if drop else df.copy()
    expected.index = idx
    result = df.set_index(keys, drop=drop, append=True)
    tm.assert_frame_equal(result, expected)