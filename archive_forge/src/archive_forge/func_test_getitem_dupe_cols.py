import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_dupe_cols(self):
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'a', 'b'])
    msg = '"None of [Index([\'baf\'], dtype='
    with pytest.raises(KeyError, match=re.escape(msg)):
        df[['baf']]