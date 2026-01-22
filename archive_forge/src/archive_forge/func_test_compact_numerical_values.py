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
@pytest.mark.parametrize('column', ['WGT', 'CYL'])
def test_compact_numerical_values(datapath, column):
    fname = datapath('io', 'sas', 'data', 'cars.sas7bdat')
    df = pd.read_sas(fname, encoding='latin-1')
    result = df[column]
    expected = df[column].round()
    tm.assert_series_equal(result, expected, check_exact=True)