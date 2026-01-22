from collections.abc import Iterator
from functools import partial
from io import (
import os
from pathlib import Path
import re
import threading
from urllib.error import URLError
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import file_path_to_url
@pytest.mark.slow
def test_banklist_header(self, banklist_data, datapath, flavor_read_html):
    from pandas.io.html import _remove_whitespace

    def try_remove_ws(x):
        try:
            return _remove_whitespace(x)
        except AttributeError:
            return x
    df = flavor_read_html(banklist_data, match='Metcalf', attrs={'id': 'table'})[0]
    ground_truth = read_csv(datapath('io', 'data', 'csv', 'banklist.csv'), converters={'Updated Date': Timestamp, 'Closing Date': Timestamp})
    assert df.shape == ground_truth.shape
    old = ['First Vietnamese American Bank In Vietnamese', 'Westernbank Puerto Rico En Espanol', 'R-G Premier Bank of Puerto Rico En Espanol', 'Eurobank En Espanol', 'Sanderson State Bank En Espanol', 'Washington Mutual Bank (Including its subsidiary Washington Mutual Bank FSB)', 'Silver State Bank En Espanol', 'AmTrade International Bank En Espanol', 'Hamilton Bank, NA En Espanol', 'The Citizens Savings Bank Pioneer Community Bank, Inc.']
    new = ['First Vietnamese American Bank', 'Westernbank Puerto Rico', 'R-G Premier Bank of Puerto Rico', 'Eurobank', 'Sanderson State Bank', 'Washington Mutual Bank', 'Silver State Bank', 'AmTrade International Bank', 'Hamilton Bank, NA', 'The Citizens Savings Bank']
    dfnew = df.map(try_remove_ws).replace(old, new)
    gtnew = ground_truth.map(try_remove_ws)
    converted = dfnew
    date_cols = ['Closing Date', 'Updated Date']
    converted[date_cols] = converted[date_cols].apply(to_datetime)
    tm.assert_frame_equal(converted, gtnew)