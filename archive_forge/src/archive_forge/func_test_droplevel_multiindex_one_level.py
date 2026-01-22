import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_droplevel_multiindex_one_level():
    index = MultiIndex.from_tuples([(2,)], names=('b',))
    result = index.droplevel([])
    expected = Index([2], name='b')
    tm.assert_index_equal(result, expected)