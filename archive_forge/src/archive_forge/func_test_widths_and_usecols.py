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
def test_widths_and_usecols():
    data = '0  1    n -0.4100.1\n0  2    p  0.2 90.1\n0  3    n -0.3140.4'
    result = read_fwf(StringIO(data), header=None, usecols=(0, 1, 3), widths=(3, 5, 1, 5, 5), index_col=False, names=('c0', 'c1', 'c3'))
    expected = DataFrame({'c0': 0, 'c1': [1, 2, 3], 'c3': [-0.4, 0.2, -0.3]})
    tm.assert_frame_equal(result, expected)