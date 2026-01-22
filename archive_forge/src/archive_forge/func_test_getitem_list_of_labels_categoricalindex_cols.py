import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_list_of_labels_categoricalindex_cols(self):
    cats = Categorical([Timestamp('12-31-1999'), Timestamp('12-31-2000')])
    expected = DataFrame([[1, 0], [0, 1]], dtype='bool', index=[0, 1], columns=cats)
    dummies = get_dummies(cats)
    result = dummies[list(dummies.columns)]
    tm.assert_frame_equal(result, expected)