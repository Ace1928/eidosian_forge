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
def test_whitespace_preservation():
    header = None
    csv_data = '\n a ,bbb\n cc,dd '
    fwf_data = '\n a bbb\n ccdd '
    result = read_fwf(StringIO(fwf_data), widths=[3, 3], header=header, skiprows=[0], delimiter='\n\t')
    expected = read_csv(StringIO(csv_data), header=header)
    tm.assert_frame_equal(result, expected)