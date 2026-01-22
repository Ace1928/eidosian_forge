from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_iloc_mask(self):
    df = DataFrame(list(range(5)), index=list('ABCDE'), columns=['a'])
    mask = df.a % 2 == 0
    msg = 'iLocation based boolean indexing cannot use an indexable as a mask'
    with pytest.raises(ValueError, match=msg):
        df.iloc[mask]
    mask.index = range(len(mask))
    msg = 'iLocation based boolean indexing on an integer type is not available'
    with pytest.raises(NotImplementedError, match=msg):
        df.iloc[mask]
    result = df.iloc[np.array([True] * len(mask), dtype=bool)]
    tm.assert_frame_equal(result, df)
    locs = np.arange(4)
    nums = 2 ** locs
    reps = [bin(num) for num in nums]
    df = DataFrame({'locs': locs, 'nums': nums}, reps)
    expected = {(None, ''): '0b1100', (None, '.loc'): '0b1100', (None, '.iloc'): '0b1100', ('index', ''): '0b11', ('index', '.loc'): '0b11', ('index', '.iloc'): 'iLocation based boolean indexing cannot use an indexable as a mask', ('locs', ''): 'Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match).', ('locs', '.loc'): 'Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match).', ('locs', '.iloc'): 'iLocation based boolean indexing on an integer type is not available'}
    for idx in [None, 'index', 'locs']:
        mask = (df.nums > 2).values
        if idx:
            mask_index = getattr(df, idx)[::-1]
            mask = Series(mask, list(mask_index))
        for method in ['', '.loc', '.iloc']:
            try:
                if method:
                    accessor = getattr(df, method[1:])
                else:
                    accessor = df
                answer = str(bin(accessor[mask]['nums'].sum()))
            except (ValueError, IndexingError, NotImplementedError) as err:
                answer = str(err)
            key = (idx, method)
            r = expected.get(key)
            if r != answer:
                raise AssertionError(f'[{key}] does not match [{answer}], received [{r}]')