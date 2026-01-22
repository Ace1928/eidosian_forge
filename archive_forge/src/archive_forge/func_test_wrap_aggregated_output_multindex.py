from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_wrap_aggregated_output_multindex(multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data.T
    df['baz', 'two'] = 'peekaboo'
    keys = [np.array([0, 0, 1]), np.array([0, 0, 1])]
    msg = re.escape('agg function failed [how->mean,dtype->')
    with pytest.raises(TypeError, match=msg):
        df.groupby(keys).agg('mean')
    agged = df.drop(columns=('baz', 'two')).groupby(keys).agg('mean')
    assert isinstance(agged.columns, MultiIndex)

    def aggfun(ser):
        if ser.name == ('foo', 'one'):
            raise TypeError('Test error message')
        return ser.sum()
    with pytest.raises(TypeError, match='Test error message'):
        df.groupby(keys).aggregate(aggfun)