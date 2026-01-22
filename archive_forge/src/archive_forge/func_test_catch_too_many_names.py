import codecs
import csv
from io import StringIO
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import PY311
from pandas.errors import (
from pandas import DataFrame
import pandas._testing as tm
@xfail_pyarrow
def test_catch_too_many_names(all_parsers):
    data = '1,2,3\n4,,6\n7,8,9\n10,11,12\n'
    parser = all_parsers
    msg = 'Too many columns specified: expected 4 and found 3' if parser.engine == 'c' else 'Number of passed names did not match number of header fields in the file'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=0, names=['a', 'b', 'c', 'd'])