import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
@pytest.mark.parametrize('klass', [tuple, list, np.array, Series, Index])
def test_cat_different_classes(klass):
    s = Series(['a', 'b', 'c'])
    result = s.str.cat(klass(['x', 'y', 'z']))
    expected = Series(['ax', 'by', 'cz'])
    tm.assert_series_equal(result, expected)