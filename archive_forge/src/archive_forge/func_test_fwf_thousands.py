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
@pytest.mark.parametrize('thousands', [',', '#', '~'])
def test_fwf_thousands(thousands):
    data = ' 1 2,334.0    5\n10   13     10.\n'
    data = data.replace(',', thousands)
    colspecs = [(0, 3), (3, 11), (12, 16)]
    expected = DataFrame([[1, 2334.0, 5], [10, 13, 10.0]])
    result = read_fwf(StringIO(data), header=None, colspecs=colspecs, thousands=thousands)
    tm.assert_almost_equal(result, expected)