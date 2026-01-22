from datetime import (
from io import StringIO
from dateutil.parser import parse as du_parse
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import parsing
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.tools.datetimes import start_caching_at
from pandas.io.parsers import read_csv
@pytest.mark.parametrize('container', [list, tuple, Index, Series])
@pytest.mark.parametrize('dim', [1, 2])
def test_concat_date_col_fail(container, dim):
    msg = 'not all elements from date_cols are numpy arrays'
    value = '19990127'
    date_cols = tuple((container([value]) for _ in range(dim)))
    with pytest.raises(ValueError, match=msg):
        parsing.concat_date_cols(date_cols)