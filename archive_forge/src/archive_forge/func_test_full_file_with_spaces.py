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
def test_full_file_with_spaces():
    test = '\nAccount                 Name  Balance     CreditLimit   AccountCreated\n101     Keanu Reeves          9315.45     10000.00           1/17/1998\n312     Gerard Butler         90.00       1000.00             8/6/2003\n868     Jennifer Love Hewitt  0           17000.00           5/25/1985\n761     Jada Pinkett-Smith    49654.87    100000.00          12/5/2006\n317     Bill Murray           789.65      5000.00             2/5/2007\n'.strip('\r\n')
    colspecs = ((0, 7), (8, 28), (30, 38), (42, 53), (56, 70))
    expected = read_fwf(StringIO(test), colspecs=colspecs)
    result = read_fwf(StringIO(test))
    tm.assert_frame_equal(result, expected)