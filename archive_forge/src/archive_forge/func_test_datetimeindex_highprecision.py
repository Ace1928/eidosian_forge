from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
@pytest.mark.parametrize('start_date', ['2017-01-01 23:59:59.999999999', '2017-01-01 23:59:59.99999999', '2017-01-01 23:59:59.9999999', '2017-01-01 23:59:59.999999', '2017-01-01 23:59:59.99999', '2017-01-01 23:59:59.9999'])
def test_datetimeindex_highprecision(self, start_date):
    s1 = Series(date_range(start=start_date, freq='D', periods=5))
    result = str(s1)
    assert start_date in result
    dti = date_range(start=start_date, freq='D', periods=5)
    s2 = Series(3, index=dti)
    result = str(s2.index)
    assert start_date in result