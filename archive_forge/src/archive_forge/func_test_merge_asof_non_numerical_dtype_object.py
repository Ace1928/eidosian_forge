import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_merge_asof_non_numerical_dtype_object():
    left = pd.DataFrame({'a': ['12', '13', '15'], 'left_val1': ['a', 'b', 'c']})
    right = pd.DataFrame({'a': ['a', 'b', 'c'], 'left_val': ['d', 'e', 'f']})
    with pytest.raises(MergeError, match='Incompatible merge dtype, .*, both sides must have numeric dtype'):
        merge_asof(left, right, left_on='left_val1', right_on='a', left_by='a', right_by='left_val')