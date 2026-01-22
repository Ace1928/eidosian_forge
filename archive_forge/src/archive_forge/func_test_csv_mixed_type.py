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
def test_csv_mixed_type(all_parsers):
    data = 'A,B,C\na,1,2\nb,3,4\nc,4,5\n'
    parser = all_parsers
    expected = DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 3, 4], 'C': [2, 4, 5]})
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)