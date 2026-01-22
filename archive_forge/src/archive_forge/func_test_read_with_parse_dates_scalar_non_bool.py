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
@pytest.mark.parametrize('kwargs', [{}, {'index_col': 'C'}])
def test_read_with_parse_dates_scalar_non_bool(all_parsers, kwargs):
    parser = all_parsers
    msg = "Only booleans, lists, and dictionaries are accepted for the 'parse_dates' parameter"
    data = 'A,B,C\n    1,2,2003-11-1'
    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), parse_dates='C', **kwargs)