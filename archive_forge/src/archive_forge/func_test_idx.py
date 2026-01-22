import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx_method', ['idxmax', 'idxmin'])
def test_idx(self, idx_method):
    df = tm.SubclassedDataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = getattr(df, idx_method)()
    assert isinstance(result, tm.SubclassedSeries)