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
@pytest.mark.parametrize('reader', ['read_csv_check_warnings', 'read_table_check_warnings'])
def test_parse_dates_date_parser_and_date_format(all_parsers, reader):
    parser = all_parsers
    data = 'Date,test\n2012-01-01,1\n,2'
    msg = "Cannot use both 'date_parser' and 'date_format'"
    with pytest.raises(TypeError, match=msg):
        getattr(parser, reader)(FutureWarning, "use 'date_format' instead", StringIO(data), parse_dates=['Date'], date_parser=pd.to_datetime, date_format='ISO8601', sep=',')