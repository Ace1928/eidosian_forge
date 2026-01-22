import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_equals_subclass(self):
    df1 = DataFrame({'a': [1, 2, 3]})
    df2 = tm.SubclassedDataFrame({'a': [1, 2, 3]})
    assert df1.equals(df2)
    assert df2.equals(df1)