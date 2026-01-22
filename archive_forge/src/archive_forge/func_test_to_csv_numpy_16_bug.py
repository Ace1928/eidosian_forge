import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def test_to_csv_numpy_16_bug(self):
    frame = DataFrame({'a': date_range('1/1/2000', periods=10)})
    buf = StringIO()
    frame.to_csv(buf)
    result = buf.getvalue()
    assert '2000-01-01' in result