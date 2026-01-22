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
def test_skip_rows_and_n_rows():
    data = 'a\tb\n1\t a\n2\t b\n3\t c\n4\t d\n5\t e\n6\t f\n    '
    result = read_fwf(StringIO(data), nrows=4, skiprows=[2, 4])
    expected = DataFrame({'a': [1, 3, 5, 6], 'b': ['a', 'c', 'e', 'f']})
    tm.assert_frame_equal(result, expected)