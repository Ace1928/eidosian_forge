from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('box2', [Series, Index, np.array, list, iter, lambda x: MultiIndex.from_arrays([x]), lambda x: x.name])
@pytest.mark.parametrize('box1', [Series, Index, np.array, list, iter, lambda x: MultiIndex.from_arrays([x]), lambda x: x.name])
@pytest.mark.parametrize('append, index_name', [(True, None), (True, 'A'), (True, 'test'), (False, None)])
@pytest.mark.parametrize('drop', [True, False])
def test_set_index_pass_arrays_duplicate(self, frame_of_index_cols, drop, append, index_name, box1, box2):
    df = frame_of_index_cols
    df.index.name = index_name
    keys = [box1(df['A']), box2(df['A'])]
    result = df.set_index(keys, drop=drop, append=append)
    keys = [box1(df['A']), box2(df['A'])]
    first_drop = False if isinstance(keys[0], str) and keys[0] == 'A' and isinstance(keys[1], str) and (keys[1] == 'A') else drop
    expected = df.set_index([keys[0]], drop=first_drop, append=append)
    expected = expected.set_index([keys[1]], drop=drop, append=True)
    tm.assert_frame_equal(result, expected)