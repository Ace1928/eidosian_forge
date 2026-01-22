import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reindex_non_unique():
    idx = MultiIndex.from_tuples([(0, 0), (1, 1), (1, 1), (2, 2)])
    a = pd.Series(np.arange(4), index=idx)
    new_idx = MultiIndex.from_tuples([(0, 0), (1, 1), (2, 2)])
    msg = 'cannot handle a non-unique multi-index!'
    with pytest.raises(ValueError, match=msg):
        a.reindex(new_idx)