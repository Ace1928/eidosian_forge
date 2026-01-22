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
def test_read_table_equivalency_to_read_csv(all_parsers):
    parser = all_parsers
    data = 'a\tb\n1\t2\n3\t4'
    expected = parser.read_csv(StringIO(data), sep='\t')
    result = parser.read_table(StringIO(data))
    tm.assert_frame_equal(result, expected)