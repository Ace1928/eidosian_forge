import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def test_to_csv_unicodewriter_quoting(self):
    df = DataFrame({'A': [1, 2, 3], 'B': ['foo', 'bar', 'baz']})
    buf = StringIO()
    df.to_csv(buf, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
    result = buf.getvalue()
    expected_rows = ['"A","B"', '1,"foo"', '2,"bar"', '3,"baz"']
    expected = tm.convert_rows_list_to_csv_str(expected_rows)
    assert result == expected