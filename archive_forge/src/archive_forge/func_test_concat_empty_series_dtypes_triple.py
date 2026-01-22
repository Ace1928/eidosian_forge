import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_empty_series_dtypes_triple(self):
    assert concat([Series(dtype='M8[ns]'), Series(dtype=np.bool_), Series(dtype=np.int64)]).dtype == np.object_