from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_cat_preserves_structure(observed, ordered):
    df = DataFrame({'Name': Categorical(['Bob', 'Greg'], ordered=ordered), 'Item': [1, 2]}, columns=['Name', 'Item'])
    expected = df.copy()
    result = df.groupby('Name', observed=observed).agg(DataFrame.sum, skipna=True).reset_index()
    tm.assert_frame_equal(result, expected)