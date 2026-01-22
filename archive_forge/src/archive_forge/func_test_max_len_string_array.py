import numpy as np
import pytest
from pandas._libs import (
from pandas.compat import IS64
from pandas import Index
import pandas._testing as tm
def test_max_len_string_array(self):
    arr = a = np.array(['foo', 'b', np.nan], dtype='object')
    assert libwriters.max_len_string_array(arr) == 3
    arr = a.astype('U').astype(object)
    assert libwriters.max_len_string_array(arr) == 3
    arr = a.astype('S').astype(object)
    assert libwriters.max_len_string_array(arr) == 3
    msg = 'No matching signature found'
    with pytest.raises(TypeError, match=msg):
        libwriters.max_len_string_array(arr.astype('U'))