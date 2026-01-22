import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_merge_asof_raise_for_duplicate_columns():
    left = pd.DataFrame([[1, 2, 'a']], columns=['a', 'a', 'left_val'])
    right = pd.DataFrame([[1, 1, 1]], columns=['a', 'a', 'right_val'])
    with pytest.raises(ValueError, match="column label 'a'"):
        merge_asof(left, right, on='a')
    with pytest.raises(ValueError, match="column label 'a'"):
        merge_asof(left, right, left_on='a', right_on='right_val')
    with pytest.raises(ValueError, match="column label 'a'"):
        merge_asof(left, right, left_on='left_val', right_on='a')