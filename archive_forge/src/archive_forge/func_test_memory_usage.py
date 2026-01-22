import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_memory_usage(self):
    df = tm.SubclassedDataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = df.memory_usage()
    assert isinstance(result, tm.SubclassedSeries)
    result = df.memory_usage(index=False)
    assert isinstance(result, tm.SubclassedSeries)