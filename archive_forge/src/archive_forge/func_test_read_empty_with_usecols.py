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
@pytest.mark.parametrize('data,kwargs,expected', [('', {}, None), ('', {'usecols': ['X']}, None), (',,', {'names': ['Dummy', 'X', 'Dummy_2'], 'usecols': ['X']}, DataFrame(columns=['X'], index=[0], dtype=np.float64)), ('', {'names': ['Dummy', 'X', 'Dummy_2'], 'usecols': ['X']}, DataFrame(columns=['X']))])
def test_read_empty_with_usecols(all_parsers, data, kwargs, expected):
    parser = all_parsers
    if expected is None:
        msg = 'No columns to parse from file'
        with pytest.raises(EmptyDataError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        result = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)