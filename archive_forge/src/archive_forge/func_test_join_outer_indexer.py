import numpy as np
import pytest
from pandas._libs.tslibs import IncompatibleFrequency
from pandas import (
import pandas._testing as tm
def test_join_outer_indexer(self):
    pi = period_range('1/1/2000', '1/20/2000', freq='D')
    result = pi._outer_indexer(pi)
    tm.assert_extension_array_equal(result[0], pi._values)
    tm.assert_numpy_array_equal(result[1], np.arange(len(pi), dtype=np.intp))
    tm.assert_numpy_array_equal(result[2], np.arange(len(pi), dtype=np.intp))