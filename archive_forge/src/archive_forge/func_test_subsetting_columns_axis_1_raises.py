import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_subsetting_columns_axis_1_raises():
    df = DataFrame({'a': [1], 'b': [2], 'c': [3]})
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby('a', axis=1)
    with pytest.raises(ValueError, match='Cannot subset columns when using axis=1'):
        gb['b']