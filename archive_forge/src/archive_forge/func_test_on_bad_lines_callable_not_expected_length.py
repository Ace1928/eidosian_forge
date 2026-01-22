from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_on_bad_lines_callable_not_expected_length(python_parser_only):
    parser = python_parser_only
    data = 'a,b\n1,2\n2,3,4,5,6\n3,4\n'
    bad_sio = StringIO(data)
    result = parser.read_csv_check_warnings(ParserWarning, 'Length of header or names', bad_sio, on_bad_lines=lambda x: x)
    expected = DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
    tm.assert_frame_equal(result, expected)