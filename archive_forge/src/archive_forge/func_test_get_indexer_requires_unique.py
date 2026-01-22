import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_indexer_requires_unique(self):
    ci = CategoricalIndex(list('aabbca'), categories=list('cab'), ordered=False)
    oidx = Index(np.array(ci))
    msg = 'Reindexing only valid with uniquely valued Index objects'
    for n in [1, 2, 5, len(ci)]:
        finder = oidx[np.random.default_rng(2).integers(0, len(ci), size=n)]
        with pytest.raises(InvalidIndexError, match=msg):
            ci.get_indexer(finder)
    for finder in [list('aabbca'), list('aababca')]:
        with pytest.raises(InvalidIndexError, match=msg):
            ci.get_indexer(finder)