from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_header_missing_rows(all_parsers):
    parser = all_parsers
    data = 'a,b\n1,2\n'
    msg = 'Passed header=\\[0,1,2\\], len of 3, but only 2 lines in file'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=[0, 1, 2])