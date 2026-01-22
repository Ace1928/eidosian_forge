import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index', [Index(np.arange(5), dtype=np.int64), RangeIndex(5)])
def test_slice_integer_frame_getitem(self, index):
    s = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), index=index)
    for idx in [slice(0.0, 1), slice(0, 1.0), slice(0.0, 1.0)]:
        result = s.loc[idx]
        indexer = slice(0, 2)
        self.check(result, s, indexer, False)
        msg = f'cannot do slice indexing on {type(index).__name__} with these indexers \\[(0|1)\\.0\\] of type float'
        with pytest.raises(TypeError, match=msg):
            s[idx]
    for idx in [slice(-10, 10), slice(-10.0, 10.0)]:
        result = s.loc[idx]
        self.check(result, s, slice(-10, 10), True)
    msg = f'cannot do slice indexing on {type(index).__name__} with these indexers \\[-10\\.0\\] of type float'
    with pytest.raises(TypeError, match=msg):
        s[slice(-10.0, 10.0)]
    for idx, res in [(slice(0.5, 1), slice(1, 2)), (slice(0, 0.5), slice(0, 1)), (slice(0.5, 1.5), slice(1, 2))]:
        result = s.loc[idx]
        self.check(result, s, res, False)
        msg = f'cannot do slice indexing on {type(index).__name__} with these indexers \\[0\\.5\\] of type float'
        with pytest.raises(TypeError, match=msg):
            s[idx]