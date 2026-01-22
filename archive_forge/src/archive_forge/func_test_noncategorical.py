import ctypes
import math
import pytest
import modin.pandas as pd
def test_noncategorical(df_from_dict):
    df = df_from_dict({'a': [1, 2, 3]})
    dfX = df.__dataframe__()
    colX = dfX.get_column_by_name('a')
    with pytest.raises(TypeError):
        colX.describe_categorical