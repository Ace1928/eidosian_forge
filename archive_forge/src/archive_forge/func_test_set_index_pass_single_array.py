from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('box', [Series, Index, np.array, list, lambda x: [list(x)], lambda x: MultiIndex.from_arrays([x])])
@pytest.mark.parametrize('append, index_name', [(True, None), (True, 'B'), (True, 'test'), (False, None)])
@pytest.mark.parametrize('drop', [True, False])
def test_set_index_pass_single_array(self, frame_of_index_cols, drop, append, index_name, box):
    df = frame_of_index_cols
    df.index.name = index_name
    key = box(df['B'])
    if box == list:
        msg = "['one', 'two', 'three', 'one', 'two']"
        with pytest.raises(KeyError, match=msg):
            df.set_index(key, drop=drop, append=append)
    else:
        name_mi = getattr(key, 'names', None)
        name = [getattr(key, 'name', None)] if name_mi is None else name_mi
        result = df.set_index(key, drop=drop, append=append)
        expected = df.set_index(['B'], drop=False, append=append)
        expected.index.names = [index_name] + name if append else name
        tm.assert_frame_equal(result, expected)