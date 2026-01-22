from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_none_delimiter(python_parser_only):
    parser = python_parser_only
    data = 'a,b,c\n0,1,2\n3,4,5,6\n7,8,9'
    expected = DataFrame({'a': [0, 7], 'b': [1, 8], 'c': [2, 9]})
    with tm.assert_produces_warning(ParserWarning, match='Skipping line 3', check_stacklevel=False):
        result = parser.read_csv(StringIO(data), header=0, sep=None, on_bad_lines='warn')
    tm.assert_frame_equal(result, expected)