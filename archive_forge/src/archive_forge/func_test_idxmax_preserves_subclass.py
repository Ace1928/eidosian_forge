import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_idxmax_preserves_subclass(self):
    df = tm.SubclassedDataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = df.idxmax()
    assert isinstance(result, tm.SubclassedSeries)