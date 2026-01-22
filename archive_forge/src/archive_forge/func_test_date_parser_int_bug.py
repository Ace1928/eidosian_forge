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
def test_date_parser_int_bug(all_parsers):
    parser = all_parsers
    data = 'posix_timestamp,elapsed,sys,user,queries,query_time,rows,accountid,userid,contactid,level,silo,method\n1343103150,0.062353,0,4,6,0.01690,3,12345,1,-1,3,invoice_InvoiceResource,search\n'
    result = parser.read_csv_check_warnings(FutureWarning, "use 'date_format' instead", StringIO(data), index_col=0, parse_dates=[0], date_parser=lambda x: datetime.fromtimestamp(int(x), tz=timezone.utc).replace(tzinfo=None), raise_on_extra_warnings=False)
    expected = DataFrame([[0.062353, 0, 4, 6, 0.0169, 3, 12345, 1, -1, 3, 'invoice_InvoiceResource', 'search']], columns=['elapsed', 'sys', 'user', 'queries', 'query_time', 'rows', 'accountid', 'userid', 'contactid', 'level', 'silo', 'method'], index=Index([Timestamp('2012-07-24 04:12:30')], name='posix_timestamp'))
    tm.assert_frame_equal(result, expected)