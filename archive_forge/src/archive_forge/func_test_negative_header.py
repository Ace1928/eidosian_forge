from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
def test_negative_header(all_parsers):
    parser = all_parsers
    data = '1,2,3,4,5\n6,7,8,9,10\n11,12,13,14,15\n'
    with pytest.raises(ValueError, match='Passing negative integer to header is invalid. For no header, use header=None instead'):
        parser.read_csv(StringIO(data), header=-1)