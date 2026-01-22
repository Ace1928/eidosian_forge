from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('agg', [{'func': {'result1': np.sum, 'result2': np.mean}}, {'A': ('result1', np.sum), 'B': ('result2', np.mean)}, {'A': NamedAgg('result1', np.sum), 'B': NamedAgg('result2', np.mean)}])
def test_agg_no_column(cases, agg):
    msg = "Column\\(s\\) \\['result1', 'result2'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        cases[['A', 'B']].agg(**agg)