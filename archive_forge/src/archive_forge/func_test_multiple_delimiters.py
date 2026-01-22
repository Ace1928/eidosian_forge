from datetime import datetime
from io import (
from pathlib import Path
import numpy as np
import pytest
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import urlopen
from pandas.io.parsers import (
def test_multiple_delimiters():
    test = '\ncol1~~~~~col2  col3++++++++++++++++++col4\n~~22.....11.0+++foo~~~~~~~~~~Keanu Reeves\n  33+++122.33\\\\\\bar.........Gerard Butler\n++44~~~~12.01   baz~~Jennifer Love Hewitt\n~~55       11+++foo++++Jada Pinkett-Smith\n..66++++++.03~~~bar           Bill Murray\n'.strip('\r\n')
    delimiter = ' +~.\\'
    colspecs = ((0, 4), (7, 13), (15, 19), (21, 41))
    expected = read_fwf(StringIO(test), colspecs=colspecs, delimiter=delimiter)
    result = read_fwf(StringIO(test), delimiter=delimiter)
    tm.assert_frame_equal(result, expected)