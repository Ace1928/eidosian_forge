import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index', [Index(list('abcde'), dtype=object), date_range('2020-01-01', periods=5), timedelta_range('1 day', periods=5), period_range('2020-01-01', periods=5)])
@pytest.mark.parametrize('idx', [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
def test_slice_non_numeric(self, index, idx, frame_or_series, indexer_sli):
    s = gen_obj(frame_or_series, index)
    if indexer_sli is tm.iloc:
        msg = f'cannot do positional indexing on {type(index).__name__} with these indexers \\[(3|4)\\.0\\] of type float'
    else:
        msg = f'cannot do slice indexing on {type(index).__name__} with these indexers \\[(3|4)(\\.0)?\\] of type (float|int)'
    with pytest.raises(TypeError, match=msg):
        indexer_sli(s)[idx]
    if indexer_sli is tm.iloc:
        msg = 'slice indices must be integers or None or have an __index__ method'
    with pytest.raises(TypeError, match=msg):
        indexer_sli(s)[idx] = 0