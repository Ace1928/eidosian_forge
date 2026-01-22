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
def test_read_csv_wrong_num_columns(all_parsers):
    data = 'A,B,C,D,E,F\n1,2,3,4,5,6\n6,7,8,9,10,11,12\n11,12,13,14,15,16\n'
    parser = all_parsers
    msg = 'Expected 6 fields in line 3, saw 7'
    if parser.engine == 'pyarrow':
        pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data))