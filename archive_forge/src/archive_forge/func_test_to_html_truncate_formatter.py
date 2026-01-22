from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_to_html_truncate_formatter(datapath):
    data = [{'A': 1, 'B': 2, 'C': 3, 'D': 4}, {'A': 5, 'B': 6, 'C': 7, 'D': 8}, {'A': 9, 'B': 10, 'C': 11, 'D': 12}, {'A': 13, 'B': 14, 'C': 15, 'D': 16}]
    df = DataFrame(data)
    fmt = lambda x: str(x) + '_mod'
    formatters = [fmt, fmt, None, None]
    result = df.to_html(formatters=formatters, max_cols=3)
    expected = expected_html(datapath, 'truncate_formatter')
    assert result == expected