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
def test_parse_multiple_delimited_dates_with_swap_warnings():
    with pytest.raises(ValueError, match='^time data "31/05/2000" doesn\\\'t match format "%m/%d/%Y", at position 1. You might want to try:'):
        pd.to_datetime(['01/01/2000', '31/05/2000', '31/05/2001', '01/02/2000'])