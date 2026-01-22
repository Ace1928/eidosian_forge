from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_on_bad_lines_callable_dont_swallow_errors(python_parser_only):
    parser = python_parser_only
    data = 'a,b\n1,2\n2,3,4,5,6\n3,4\n'
    bad_sio = StringIO(data)
    msg = 'This function is buggy.'

    def bad_line_func(bad_line):
        raise ValueError(msg)
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(bad_sio, on_bad_lines=bad_line_func)