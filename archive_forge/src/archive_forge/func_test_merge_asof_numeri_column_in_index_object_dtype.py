import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_merge_asof_numeri_column_in_index_object_dtype():
    left = pd.DataFrame({'b': [10, 11, 12]}, index=Index(['1', '2', '3'], name='a'))
    right = pd.DataFrame({'c': [20, 21, 22]}, index=Index(['m', 'n', 'o'], name='a'))
    with pytest.raises(MergeError, match='Incompatible merge dtype, .*, both sides must have numeric dtype'):
        merge_asof(left, right, left_on='a', right_on='a')
    left = left.reset_index().set_index(['a', 'b'])
    right = right.reset_index().set_index(['a', 'c'])
    with pytest.raises(MergeError, match='Incompatible merge dtype, .*, both sides must have numeric dtype'):
        merge_asof(left, right, left_on='a', right_on='a')