import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.parametrize('idx', [[0, 1, 2], pd.array([0, 1, 2], dtype='Int64'), np.array([0, 1, 2])], ids=['list', 'integer-array', 'numpy-array'])
def test_setitem_integer_array(self, data, idx, box_in_series, request):
    if box_in_series:
        mark = pytest.mark.xfail(reason='cannot set using a list-like indexer with a different length')
        request.applymarker(mark)
    super().test_setitem_integer_array(data, idx, box_in_series)