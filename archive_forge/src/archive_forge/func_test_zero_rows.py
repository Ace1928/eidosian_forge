import contextlib
from datetime import datetime
import io
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import EmptyDataError
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sas7bdat import SAS7BDATReader
def test_zero_rows(datapath):
    fname = datapath('io', 'sas', 'data', 'zero_rows.sas7bdat')
    result = pd.read_sas(fname)
    expected = pd.DataFrame([{'char_field': 'a', 'num_field': 1.0}]).iloc[:0]
    tm.assert_frame_equal(result, expected)