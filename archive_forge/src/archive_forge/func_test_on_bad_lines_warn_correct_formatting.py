import codecs
import csv
from io import StringIO
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import PY311
from pandas.errors import (
from pandas import DataFrame
import pandas._testing as tm
def test_on_bad_lines_warn_correct_formatting(all_parsers):
    parser = all_parsers
    data = '1,2\na,b\na,b,c\na,b,d\na,b\n'
    expected = DataFrame({'1': 'a', '2': ['b'] * 2})
    match_msg = 'Skipping line'
    expected_warning = ParserWarning
    if parser.engine == 'pyarrow':
        match_msg = 'Expected 2 columns, but found 3: a,b,c'
        expected_warning = (ParserWarning, DeprecationWarning)
    with tm.assert_produces_warning(expected_warning, match=match_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), on_bad_lines='warn')
    tm.assert_frame_equal(result, expected)