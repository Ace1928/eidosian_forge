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
def test_bytes_io_input():
    data = BytesIO('שלום\nשלום'.encode())
    result = read_fwf(data, widths=[2, 2], encoding='utf8')
    expected = DataFrame([['של', 'ום']], columns=['של', 'ום'])
    tm.assert_frame_equal(result, expected)