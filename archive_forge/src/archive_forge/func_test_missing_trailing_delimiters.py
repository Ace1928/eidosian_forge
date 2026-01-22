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
def test_missing_trailing_delimiters(all_parsers):
    parser = all_parsers
    data = 'A,B,C,D\n1,2,3,4\n1,3,3,\n1,4,5'
    result = parser.read_csv(StringIO(data))
    expected = DataFrame([[1, 2, 3, 4], [1, 3, 3, np.nan], [1, 4, 5, np.nan]], columns=['A', 'B', 'C', 'D'])
    tm.assert_frame_equal(result, expected)