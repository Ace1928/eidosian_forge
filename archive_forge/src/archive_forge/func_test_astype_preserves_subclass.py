import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_preserves_subclass(self):
    df = tm.SubclassedDataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = df.astype({'A': np.int64, 'B': np.int32, 'C': np.float64})
    assert isinstance(result, tm.SubclassedDataFrame)