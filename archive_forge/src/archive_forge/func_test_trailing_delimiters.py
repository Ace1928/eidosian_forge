from datetime import datetime
from inspect import signature
from io import StringIO
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
@skip_pyarrow
def test_trailing_delimiters(all_parsers):
    data = 'A,B,C\n1,2,3,\n4,5,6,\n7,8,9,'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=False)
    expected = DataFrame({'A': [1, 4, 7], 'B': [2, 5, 8], 'C': [3, 6, 9]})
    tm.assert_frame_equal(result, expected)