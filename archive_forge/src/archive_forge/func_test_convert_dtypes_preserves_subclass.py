import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_convert_dtypes_preserves_subclass(self, gpd_style_subclass_df):
    df = tm.SubclassedDataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = df.convert_dtypes()
    assert isinstance(result, tm.SubclassedDataFrame)
    result = gpd_style_subclass_df.convert_dtypes()
    assert isinstance(result, type(gpd_style_subclass_df))