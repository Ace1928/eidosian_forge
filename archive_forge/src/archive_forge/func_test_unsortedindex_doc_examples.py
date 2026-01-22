import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.frozen import FrozenList
def test_unsortedindex_doc_examples():
    dfm = DataFrame({'jim': [0, 0, 1, 1], 'joe': ['x', 'x', 'z', 'y'], 'jolie': np.random.default_rng(2).random(4)})
    dfm = dfm.set_index(['jim', 'joe'])
    with tm.assert_produces_warning(PerformanceWarning):
        dfm.loc[1, 'z']
    msg = 'Key length \\(2\\) was greater than MultiIndex lexsort depth \\(1\\)'
    with pytest.raises(UnsortedIndexError, match=msg):
        dfm.loc[(0, 'y'):(1, 'z')]
    assert not dfm.index._is_lexsorted()
    assert dfm.index._lexsort_depth == 1
    dfm = dfm.sort_index()
    dfm.loc[1, 'z']
    dfm.loc[(0, 'y'):(1, 'z')]
    assert dfm.index._is_lexsorted()
    assert dfm.index._lexsort_depth == 2