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
def test_short_single_line(all_parsers):
    parser = all_parsers
    columns = ['a', 'b', 'c']
    data = '1,2'
    result = parser.read_csv(StringIO(data), header=None, names=columns)
    expected = DataFrame({'a': [1], 'b': [2], 'c': [np.nan]})
    tm.assert_frame_equal(result, expected)