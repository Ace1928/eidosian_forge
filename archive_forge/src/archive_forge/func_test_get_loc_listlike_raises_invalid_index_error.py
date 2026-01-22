import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas._testing as tm
def test_get_loc_listlike_raises_invalid_index_error(self, index):
    key = np.array([0, 1], dtype=np.intp)
    with pytest.raises(InvalidIndexError, match='\\[0 1\\]'):
        index.get_loc(key)
    with pytest.raises(InvalidIndexError, match='\\[False  True\\]'):
        index.get_loc(key.astype(bool))