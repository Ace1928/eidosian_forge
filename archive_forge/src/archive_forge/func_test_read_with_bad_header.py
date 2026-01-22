from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_read_with_bad_header(all_parsers):
    parser = all_parsers
    msg = 'but only \\d+ lines in file'
    with pytest.raises(ValueError, match=msg):
        s = StringIO(',,')
        parser.read_csv(s, header=[10])