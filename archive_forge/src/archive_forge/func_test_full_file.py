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
def test_full_file():
    test = 'index                             A    B    C\n2000-01-03T00:00:00  0.980268513777    3  foo\n2000-01-04T00:00:00  1.04791624281    -4  bar\n2000-01-05T00:00:00  0.498580885705   73  baz\n2000-01-06T00:00:00  1.12020151869     1  foo\n2000-01-07T00:00:00  0.487094399463    0  bar\n2000-01-10T00:00:00  0.836648671666    2  baz\n2000-01-11T00:00:00  0.157160753327   34  foo'
    colspecs = ((0, 19), (21, 35), (38, 40), (42, 45))
    expected = read_fwf(StringIO(test), colspecs=colspecs)
    result = read_fwf(StringIO(test))
    tm.assert_frame_equal(result, expected)