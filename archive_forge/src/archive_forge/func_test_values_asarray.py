import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_values_asarray(self, arr_data, arr):
    tm.assert_almost_equal(arr.to_dense(), arr_data)