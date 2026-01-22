import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_empty_series_dtype_category_with_array(self):
    assert concat([Series(np.array([]), dtype='category'), Series(dtype='float64')]).dtype == 'float64'