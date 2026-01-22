from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('box', [list, np.array, Index, Series])
def test_getitem_no_matches(self, box):
    ser = Series(['A', 'B'])
    key = Series(['C'], dtype=object)
    key = box(key)
    msg = "None of \\[Index\\(\\['C'\\], dtype='object|string'\\)\\] are in the \\[index\\]"
    with pytest.raises(KeyError, match=msg):
        ser[key]